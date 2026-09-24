import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    add,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import RR_FEATURES, build_full_dataset, classes
from helpers import (build_representative_dataset, evaluate_tflite, normalize_beats, print_results,
                     smoothed_class_weights)


output_directory = Path('models')

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Compact 1D pre-activation ResNet.
#
# Changes compared to the previous version, all aimed at fitting on an MCU:
#   - fewer filters (16 -> 64 instead of 32 -> 256)
#   - smaller kernel (7 instead of 16)
#   - fewer residual blocks (6 instead of 15)
#   - 1x1 convolution shortcut instead of MaxPooling + Lambda(zeropad),
#     which avoids custom layers and extra TFLite ops (ZEROS_LIKE, CONCATENATION)
#   - GlobalAveragePooling1D before the classifier
#   - a second input with the beat's RR intervals (load_data.RR_FEATURES),
#     joined to the pooled morphology features: the waveform alone doesn't
#     show that a beat came early, which is what separates S from N
#   - each beat is standardized (zero mean, unit variance) before the
#     network, so amplitude differences between patients/electrodes don't
#     dominate (normalize_beats, and the same step in tinyml_app_cnn.cc)
#
# Resulting model: roughly 100-150k parameters, ~150 KB after int8 quantization
# (previous version: ~8M parameters, ~32 MB in float32).
# ---------------------------------------------------------------------------

def conv_layer(filters, kernel_size, strides=1):
    return Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        strides=strides,
        kernel_initializer='he_normal',
    )


def first_conv_block(inputs, config):
    layer = conv_layer(config.base_filters, config.kernel_size)(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    shortcut = layer

    layer = conv_layer(config.base_filters, config.kernel_size)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(config.drop_rate)(layer)
    layer = conv_layer(config.base_filters, config.kernel_size)(layer)
    return add([shortcut, layer])


def residual_block(layer, filters, strides, config):
    in_filters = layer.shape[-1]

    # Projection shortcut when the shape changes, identity otherwise.
    if strides != 1 or in_filters != filters:
        shortcut = Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            strides=strides,
            kernel_initializer='he_normal',
        )(layer)
    else:
        shortcut = layer

    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = conv_layer(filters, config.kernel_size, strides)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(config.drop_rate)(layer)
    layer = conv_layer(filters, config.kernel_size)(layer)
    return add([shortcut, layer])


def main_loop_blocks(layer, config):
    filters = config.base_filters
    for block_index in range(config.n_blocks):
        # Downsample and widen every other block: 256 -> 128 -> 64 -> 32 samples.
        if block_index % 2 == 0:
            strides = 2
            if block_index > 0:
                filters *= 2
        else:
            strides = 1
        layer = residual_block(layer, filters, strides, config)
    return layer


def output_block(layer, inputs, rr_input, config):
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = concatenate([layer, rr_input])
    layer = Dense(config.dense_units, activation='relu')(layer)
    outputs = Dense(len(classes), activation='softmax')(layer)
    # The .tflite orders its inputs by name, not in this order:
    # tinyml_app_cnn.cc and evaluate_tflite() find each input by size, and
    # the calibration data is fed by name (export_model).
    model = Model(inputs=[inputs, rr_input], outputs=outputs, name='cnn_classifier')

    model.compile(
        # The previous learning rate (0.1) is far too high for Adam.
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def cnn_model(config):
    print('[MODEL] Building CNN model...')
    inputs = Input(shape=(config.input_size, 1), name='input')
    rr_input = Input(shape=(len(RR_FEATURES),), name='rr')
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    model = output_block(layer, inputs, rr_input, config)
    print(f'[MODEL] CNN model built and compiled ({model.count_params():,} parameters).')
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, train_inputs):
    print('[EXPORT] Starting model export...')
    output_directory.mkdir(parents=True, exist_ok=True)

    keras_path = output_directory / 'cnn_classifier.keras'
    tflite_path = output_directory / 'cnn_classifier.tflite'
    model.save(keras_path)

    # OPIMIZATIONS
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset(
        {model_input.name: array for model_input, array in zip(model.inputs, train_inputs)})
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f'Keras model saved to {keras_path}')
    print(f'TFLite model saved to {tflite_path} ({size_kb:.1f} KB)')
    print('[EXPORT] Model export completed.')
    return tflite_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def prepare_inputs(dataset):
    """Model inputs ([beats, rr], in the model's input order) and labels."""
    beats = np.expand_dims(normalize_beats(dataset.X), axis=2)
    rr = np.asarray(dataset.rr, dtype=np.float32)
    if np.any(np.isnan(beats)) or np.any(np.isnan(rr)) or np.any(np.isnan(dataset.y)):
        raise ValueError('Data contains None/NaN values')
    return [beats, rr], np.asarray(dataset.y).astype(int)


def build_training_callbacks(config):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=config.patience,
            mode='min',
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=config.min_lr,
            mode='min',
            verbose=1,
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        ),
        ModelCheckpoint(
            str(output_directory / 'cnn_classifier.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1,
        ),
    ]


def cnn_train(config, train, validation, test):
    print('[TRAIN] Starting CNN training pipeline...')
    train_inputs, y = prepare_inputs(train)
    validation_inputs, yval = prepare_inputs(validation)
    test_inputs, ytest = prepare_inputs(test)
    print('Train shapes:', [a.shape for a in train_inputs], 'y:', y.shape)
    print('Validation shapes:', [a.shape for a in validation_inputs], 'y:', yval.shape)
    print('Test shapes:', [a.shape for a in test_inputs], 'y:', ytest.shape)

    output_directory.mkdir(parents=True, exist_ok=True)

    if config.export_only:
        # Skips training and re-exports the model already trained, e.g. to
        # regenerate the .tflite after a converter change.
        print(f'[MODEL] export_only: loading trained model from {config.trained_model}')
        model = keras.models.load_model(config.trained_model)
        export_and_evaluate(config, model, train_inputs, test_inputs, ytest)
        return

    if config.checkpoint_path is not None:
        print(f'[MODEL] Loading checkpoint from: {config.checkpoint_path}')
        model = keras.models.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch
    else:
        model = cnn_model(config)
        initial_epoch = 0

    class_weight = smoothed_class_weights(y)
    print(f'[TRAIN] Class weights: {class_weight}')

    print('[TRAIN] Starting model.fit...')
    model.fit(
        train_inputs,
        y,
        validation_data=(validation_inputs, yval),
        epochs=config.epochs,
        batch_size=config.batch,
        callbacks=build_training_callbacks(config),
        initial_epoch=initial_epoch,
        verbose=1,
        class_weight=class_weight,
    )
    print('[TRAIN] Training completed.')

    export_and_evaluate(config, model, train_inputs, test_inputs, ytest)


def export_and_evaluate(config, model, train_inputs, test_inputs, ytest):
    # Final metrics on the test set (DS2): patients never seen in training
    # or in the early stopping / model selection done on the validation set.
    tflite_path = export_model(model, train_inputs)
    evaluate_tflite(tflite_path, test_inputs, ytest)

    print('[EVALUATION] Starting test evaluation (Keras float model)...')
    print_results(config, model, test_inputs, ytest, classes)
    print('[EVALUATION] Test evaluation completed.')


def main():
    print('[START] CNN classifier execution started.')
    config = SimpleNamespace(
        split=True,
        input_size=256,
        base_filters=16,
        kernel_size=7,
        n_blocks=6,
        drop_rate=0.2,
        dense_units=32,
        feature='MLII',
        epochs=80,
        batch=256,
        learning_rate=1e-3,
        patience=10,
        min_lr=5e-5,
        checkpoint_path=None,
        resume_epoch=0,
        trained_model=str(output_directory / 'cnn_classifier.keras'),
        export_only=False,  # True: skip training and re-export trained_model
    )
    print('[CONFIG] CNN configuration:')
    print(vars(config))
    print('[DATA] Loading ECG dataset...')
    train, validation, test = build_full_dataset(config)
    print('[DATA] ECG dataset loaded successfully.')
    cnn_train(config, train, validation, test)
    print('[DONE] CNN classifier execution finished.')


if __name__ == '__main__':
    main()