import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    DepthwiseConv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Multiply,
    add,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import RR_FEATURES, build_full_dataset, classes
from helpers import (build_representative_dataset, evaluate_tflite, macro_f1, print_results,
                     smoothed_class_weights)


MODEL_NAME = 'cnn_classifier_mobilenet_v3'
output_directory = Path('models')

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# MobileNet 1D (V3-style)
#
# Changes compared to the V2 version:
#   - hard-swish activation in the stem, the deeper blocks and the head
#     (ReLU kept in the early, high-resolution blocks, as in the paper)
#   - Squeeze-and-Excitation (SE) after the depthwise conv in the deeper
#     blocks, with hard-sigmoid gating and squeeze = exp_channels / 4
#     rounded to a multiple of 8
#   - per-block table (kernel, expansion channels, output channels, SE,
#     activation, stride) instead of a fixed expansion factor
#   - residual only when stride == 1 and channels match (no 1x1
#     projection shortcut, as in the original MobileNetV3)
#   - "efficient last stage": 1x1 conv + hard-swish before the global
#     pooling, hard-swish dense layer after it, dropout only before the
#     classifier
#
# Kept from the V2 version:
#   - RR features through their own branch (BatchNormalization + Dense)
#     before being joined to the pooled morphology features
#   - per-beat standardization (normalize_beats / model_cnn.cc)
# ---------------------------------------------------------------------------

# (kernel, exp_channels, out_channels, use_se, activation, stride)
# Expansion 2x (exp = 2 * in, as in the V2 version) instead of the paper's
# ~3-6x, so the model stays close to the V2 version in size and MACs
# (~69k weights / ~2.3M MACs vs ~40k / ~1.9M): the comparison measures the
# V3 changes (SE, hard-swish, kernels), not a bigger network.
BNECK_CONFIG = [
    (7, 16,  16, True,  'relu',   1),   # 256 (no expansion: exp == in)
    (7, 32,  16, False, 'relu',   2),   # 256 -> 128
    (7, 32,  16, False, 'relu',   1),
    (5, 64,  32, True,  'hswish', 2),   # 128 -> 64
    (5, 64,  32, True,  'hswish', 1),
    (5, 128, 64, True,  'hswish', 2),   # 64 -> 32
    (5, 128, 64, True,  'hswish', 1),
]


def make_divisible(value, divisor=8, min_value=None):
    """Rounds a channel count to a multiple of divisor (as in MobileNetV3)."""
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def apply_activation(x, kind):
    if kind == 'hswish':
        return Activation('hard_silu')(x)  # x * relu6(x + 3) / 6 (alias: hard_swish)
    if kind == 'relu':
        return Activation('relu')(x)
    raise ValueError(f'Unknown activation: {kind}')


def squeeze_excite(x, ratio=0.25):
    """Squeeze-and-Excitation: reweights each channel by a gate computed
    from the channel's global average."""
    channels = x.shape[-1]
    squeeze = make_divisible(channels * ratio)
    s = GlobalAveragePooling1D(keepdims=True)(x)
    s = Conv1D(squeeze, kernel_size=1, padding='same')(s)
    s = Activation('relu')(s)
    s = Conv1D(channels, kernel_size=1, padding='same')(s)
    s = Activation('hard_sigmoid')(s)  # relu6(x + 3) / 6
    return Multiply()([x, s])


def bneck_block(layer, kernel_size, exp_channels, out_channels, use_se, activation, strides):
    """MobileNetV3 bottleneck: expand (1x1) -> depthwise -> SE -> project (1x1, linear)."""
    in_channels = layer.shape[-1]
    x = layer

    if exp_channels != in_channels:
        x = Conv1D(exp_channels, kernel_size=1, padding='same', use_bias=False,
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = apply_activation(x, activation)

    x = DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding='same',
                        use_bias=False, depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = apply_activation(x, activation)

    if use_se:
        x = squeeze_excite(x)

    x = Conv1D(out_channels, kernel_size=1, padding='same', use_bias=False,
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    if strides == 1 and in_channels == out_channels:
        x = add([layer, x])
    return x


def stem(inputs, config):
    x = Conv1D(config.base_filters, kernel_size=config.kernel_size, padding='same', strides=1,
               use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    return apply_activation(x, 'hswish')


def main_loop_blocks(layer, config):
    for kernel_size, exp_channels, out_channels, use_se, activation, strides in config.bneck:
        layer = bneck_block(layer, kernel_size, exp_channels, out_channels,
                            use_se, activation, strides)
    return layer


def output_block(layer, inputs, rr_input, config):
    # Efficient last stage: widen with a 1x1 conv before pooling
    layer = Conv1D(config.last_channels, kernel_size=1, padding='same', use_bias=False,
                   kernel_initializer='he_normal')(layer)
    layer = BatchNormalization()(layer)
    layer = apply_activation(layer, 'hswish')
    layer = GlobalAveragePooling1D()(layer)

    # RR branch
    rr_layer = BatchNormalization()(rr_input)
    rr_layer = Dense(config.rr_units, activation='relu')(rr_layer)
    layer = concatenate([layer, rr_layer])

    layer = Dense(config.dense_units)(layer)
    layer = apply_activation(layer, 'hswish')
    if config.drop_rate > 0:
        layer = Dropout(config.drop_rate)(layer)
    outputs = Dense(len(classes), activation='softmax')(layer)

    model = Model(inputs=[inputs, rr_input], outputs=outputs, name=MODEL_NAME)

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def cnn_model(config):
    print('[MODEL] Building MobileNetV3-1D model...')
    inputs = Input(shape=(config.input_size, 1), name='input')
    rr_input = Input(shape=(len(RR_FEATURES),), name='rr')

    layer = stem(inputs, config)
    layer = main_loop_blocks(layer, config)
    model = output_block(layer, inputs, rr_input, config)

    print(f'[MODEL] MobileNetV3 model built and compiled ({model.count_params():,} parameters).')
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, train_inputs):
    print('[EXPORT] Starting model export...')
    output_directory.mkdir(parents=True, exist_ok=True)

    keras_path = output_directory / f'{MODEL_NAME}.keras'
    tflite_path = output_directory / f'{MODEL_NAME}.tflite'
    model.save(keras_path)

    # OPTIMIZATIONS
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

    # Lists the TFLite ops, to register them in the op resolver of model_cnn.cc
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

    print('[EXPORT] Model export completed.')
    return tflite_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def normalize_beats(X):
    """Standardizes each beat to zero mean and unit variance. Must match
    normalize_beat() in model_cnn.cc."""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-6)


def prepare_inputs(dataset):
    beats = np.expand_dims(normalize_beats(dataset.X), axis=2)
    rr = np.asarray(dataset.rr, dtype=np.float32)
    if np.any(np.isnan(beats)) or np.any(np.isnan(rr)) or np.any(np.isnan(dataset.y)):
        raise ValueError('Data contains None/NaN values')
    return [beats, rr], np.asarray(dataset.y).astype(int)


class ValidationMacroF1(Callback):
    """Adds val_macro_f1 (helpers.macro_f1 over N, S, V, F) to the epoch
    logs, so early stopping and the checkpoint keep the epoch that best
    separates the classes on the validation patients. val_loss follows the
    class-weighted loss, and accuracy is dominated by N (always answering N
    already scores ~0.89). Must come before the callbacks that read it."""

    def __init__(self, inputs, y):
        super().__init__()
        self.inputs = inputs
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        predictions = np.argmax(self.model.predict(self.inputs, batch_size=1024, verbose=0), axis=1)
        score = macro_f1(self.y, predictions)
        if logs is not None:
            logs['val_macro_f1'] = score
        print(f'[TRAIN] Epoch {epoch + 1}: val_macro_f1 = {score:.4f}')


def build_training_callbacks(config, validation_inputs, yval):
    return [
        ValidationMacroF1(validation_inputs, yval),
        EarlyStopping(
            monitor='val_macro_f1',
            patience=config.patience,
            mode='max',
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
            str(output_directory / f'{MODEL_NAME}.keras'),
            monitor='val_macro_f1',
            mode='max',
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
        export_and_evaluate(config, model, train_inputs, test_inputs, ytest, test.records)
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
        callbacks=build_training_callbacks(config, validation_inputs, yval),
        initial_epoch=initial_epoch,
        verbose=1,
        class_weight=class_weight,
    )
    print('[TRAIN] Training completed.')

    export_and_evaluate(config, model, train_inputs, test_inputs, ytest, test.records)


def export_and_evaluate(config, model, train_inputs, test_inputs, ytest, test_records):
    # Final metrics on the test set (DS2): patients never seen in training
    # or in the early stopping / model selection done on the validation set.
    tflite_path = export_model(model, train_inputs)
    evaluate_tflite(tflite_path, test_inputs, ytest, test_records)

    print('[EVALUATION] Starting test evaluation (Keras float model)...')
    print_results(config, model, test_inputs, ytest, classes, test_records)
    print('[EVALUATION] Test evaluation completed.')


def main():
    print('[START] CNN classifier execution started.')
    config = SimpleNamespace(
        split=True,
        input_size=256,
        base_filters=16,
        kernel_size=7,          # stem kernel; block kernels come from bneck
        bneck=BNECK_CONFIG,
        last_channels=64,       # 1x1 conv before pooling (128 in the paper's proportion)
        drop_rate=0.2,          # now only before the classifier
        dense_units=32,
        rr_units=16,
        feature='MLII',
        epochs=80,
        batch=256,
        learning_rate=1e-3,
        patience=10,
        min_lr=5e-5,
        checkpoint_path=None,
        resume_epoch=0,
        trained_model=str(output_directory / f'{MODEL_NAME}.keras'),
        export_only=False,
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
