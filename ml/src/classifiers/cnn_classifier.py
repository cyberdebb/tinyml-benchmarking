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
)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset, classes
from helpers import print_results

OUTPUT_DIRECTORY = Path('models')

# Flash budget for the model on the ESP32-S3 (the whole app partition is 7 MB,
# and the TFLite Micro runtime itself also needs space).
MAX_TFLITE_KB = 1024


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


def output_block(layer, inputs, config):
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling1D()(layer)
    outputs = Dense(len(classes), activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=outputs, name='cnn_classifier')

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
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    model = output_block(layer, inputs, config)
    print(f'[MODEL] CNN model built and compiled ({model.count_params():,} parameters).')
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_representative_dataset(Xe, n_samples=500, seed=1):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(Xe), size=min(n_samples, len(Xe)), replace=False)

    def representative_dataset():
        for index in indices:
            yield [Xe[index:index + 1].astype(np.float32)]

    return representative_dataset


def export_model(model, Xe_train):
    print('[EXPORT] Starting model export...')
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    keras_path = OUTPUT_DIRECTORY / 'cnn_classifier.keras'
    tflite_path = OUTPUT_DIRECTORY / 'cnn_classifier.tflite'
    model.save(keras_path)

    # Full integer (int8) quantization: weights, activations, input and output.
    # The ESP32 firmware already quantizes the input and dequantizes the output.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset(Xe_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f'Keras model saved to {keras_path}')
    print(f'TFLite model saved to {tflite_path} ({size_kb:.1f} KB)')
    if size_kb > MAX_TFLITE_KB:
        print(f'[WARNING] TFLite model is larger than {MAX_TFLITE_KB} KB and may not fit on the ESP32-S3.')
    print('[EXPORT] Model export completed.')
    return tflite_path


def evaluate_tflite(tflite_path, Xe, y):
    """Runs the quantized model on the validation set to check the accuracy
    after quantization (this is what actually runs on the ESP32)."""
    print('[EVALUATION] Evaluating quantized TFLite model...')
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    predictions = np.empty(len(Xe), dtype=np.int64)

    for index in range(len(Xe)):
        sample = Xe[index:index + 1].astype(np.float32)
        if input_details['dtype'] == np.int8:
            sample = np.clip(np.round(sample / input_scale + input_zero_point), -128, 127)
            sample = sample.astype(np.int8)
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        predictions[index] = np.argmax(interpreter.get_tensor(output_details['index'])[0])

    accuracy = np.mean(predictions == y)
    print(f'[EVALUATION] TFLite int8 accuracy: {accuracy:.4f}')

    ops = sorted({op['op_name'] for op in interpreter._get_ops_details()})
    print(f'[EVALUATION] TFLite ops used (must be registered in the firmware): {ops}')
    return accuracy


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def prepare_training_data(config, X, y, Xval, yval):
    print('[DATA] Preparing training and validation data...')
    Xe = np.expand_dims(X, axis=2)
    if not config.split:
        from sklearn.model_selection import train_test_split

        prepared_data = train_test_split(Xe, y, test_size=0.2, random_state=1)
        print('[DATA] Split created from the training data.')
        return prepared_data

    Xvale = np.expand_dims(Xval, axis=2)
    print('Train shapes - Xe:', Xe.shape, 'y:', y.shape)
    print('Validation shapes - Xvale:', Xvale.shape, 'yval:', yval.shape)
    print('[DATA] Data preparation completed.')
    return Xe, Xvale, y, yval


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
            str(OUTPUT_DIRECTORY / 'cnn_classifier.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1,
        ),
    ]


def cnn_train(config, X, y, Xval=None, yval=None):
    print('[TRAIN] Starting CNN training pipeline...')
    print('Initial shapes - X:', X.shape, 'y:', y.shape)
    print('Any NaN in initial X:', np.any(np.isnan(X)), 'y:', np.any(np.isnan(y)))

    Xe, Xvale, y, yval = prepare_training_data(config, X, y, Xval, yval)

    if np.any(np.isnan(Xe)) or np.any(np.isnan(y)):
        raise ValueError('Input data contains None/NaN values')
    if np.any(np.isnan(Xvale)) or np.any(np.isnan(yval)):
        raise ValueError('Validation data contains None/NaN values')

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if config.checkpoint_path is not None:
        print(f'[MODEL] Loading checkpoint from: {config.checkpoint_path}')
        model = keras.models.load_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch
    else:
        model = cnn_model(config)
        initial_epoch = 0

    unique_classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
    class_weight = dict(zip(unique_classes, class_weights))
    print(f'[TRAIN] Class weights: {class_weight}')

    print('[TRAIN] Starting model.fit...')
    model.fit(
        Xe,
        y,
        validation_data=(Xvale, yval),
        epochs=config.epochs,
        batch_size=config.batch,
        callbacks=build_training_callbacks(config),
        initial_epoch=initial_epoch,
        verbose=1,
        class_weight=class_weight,
    )
    print('[TRAIN] Training completed.')

    tflite_path = export_model(model, Xe)
    evaluate_tflite(tflite_path, Xvale, yval)
    
    print('[EVALUATION] Starting validation evaluation (Keras float model)...')
    print_results(config, model, Xvale, yval, classes)
    print('[EVALUATION] Validation evaluation completed.')


def main():
    print('[START] CNN classifier execution started.')
    config = SimpleNamespace(
        split=True,
        input_size=256,
        base_filters=16,
        kernel_size=7,
        n_blocks=6,
        drop_rate=0.2,
        feature='MLII',
        epochs=80,
        batch=256,
        learning_rate=1e-3,
        patience=10,
        min_lr=5e-5,
        checkpoint_path=None,
        resume_epoch=0,
        trained_model=str(OUTPUT_DIRECTORY / 'cnn_classifier.keras'),
        export_only=True,  # set to False to train again
    )
    print('[CONFIG] CNN configuration:')
    print(vars(config))
    print('[DATA] Loading ECG dataset...')
    X, y, Xval, yval = build_full_dataset(config)
    print('[DATA] ECG dataset loaded successfully.')
    cnn_train(config, X, y, Xval, yval)
    print('[DONE] CNN classifier execution finished.')


if __name__ == '__main__':
    main()