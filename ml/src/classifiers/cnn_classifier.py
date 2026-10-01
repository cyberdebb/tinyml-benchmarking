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
    add,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import RR_FEATURES, build_full_dataset, classes
from helpers import (build_representative_dataset, evaluate_tflite, macro_f1, print_results,
                     smoothed_class_weights)


output_directory = Path('models')

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# MobileNet 1D
#
# Changes compared to the previous version, all aimed at fitting on an MCU:
#   - fewer filters (16 -> 64 instead of 32 -> 256)
#   - smaller kernel (7 instead of 16)
#   - fewer residual blocks (6 instead of 15)
#   - 1x1 convolution shortcut instead of MaxPooling + Lambda(zeropad),
#     which avoids custom layers and extra TFLite ops (ZEROS_LIKE, CONCATENATION)
#   - GlobalAveragePooling1D before the classifier
#   - a second input with the beat's RR features (load_data.RR_FEATURES,
#     log ratios around 0): the waveform alone doesn't show that a beat
#     came early, which is what separates S from N. The RR features go
#     through their own small branch (BatchNormalization + Dense, see
#     output_block) before being joined to the pooled morphology features:
#     joined raw, 4 values next to 64 learned ones, the network mostly
#     ignored them (S sensitivity ~17% vs ~63% for the MLP on the same RR
#     features)
#   - each beat is standardized (zero mean, unit variance) before the
#     network, so amplitude differences between patients/electrodes don't
#     dominate (normalize_beats, and the same step in model_cnn.cc)
#
# Resulting model: roughly 100-150k parameters, ~150 KB after int8 quantization
# (previous version: ~8M parameters, ~32 MB in float32).
# ---------------------------------------------------------------------------

def mobilenet_block(layer, filters, strides, config, expansion=2):
    """
    Inverted Residual Block (baseado no MobileNetV2).
    Utiliza Depthwise Separable Convolutions para reduzir drasticamente os parâmetros.
    """
    in_filters = layer.shape[-1]
    
    # 1. Expansion phase (Pointwise 1x1) - expande o número de canais internos
    if expansion > 1:
        x = Conv1D(
            filters=in_filters * expansion,
            kernel_size=1,
            padding='same',
            use_bias=False, # Não precisamos de bias antes do BatchNorm
            kernel_initializer='he_normal'
        )(layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = layer

    # 2. Depthwise Convolution - aplica filtros espacialmente por canal
    x = DepthwiseConv1D(
        kernel_size=config.kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        depthwise_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if config.drop_rate > 0:
        x = Dropout(config.drop_rate)(x)

    # 3. Projection phase (Pointwise 1x1) - Linear Bottleneck (Sem ReLU no final)
    x = Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = BatchNormalization()(x)

    # Residual Connection (apenas se a resolução temporal e número de canais baterem)
    if strides == 1 and in_filters == filters:
        shortcut = layer
        return add([shortcut, x])
    else:
        # Se os canais mudaram, fazemos um atalho 1x1 linear para igualar o shape
        shortcut = Conv1D(
            filters=filters,
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )(layer)
        shortcut = BatchNormalization()(shortcut)
        return add([shortcut, x])


def first_conv_block(inputs, config):
    """
    A primeira camada é uma Conv1D padrão para extrair características iniciais,
    seguida do primeiro bloco MobileNet (geralmente sem expansão para economizar recursos).
    """
    layer = Conv1D(
        filters=config.base_filters, 
        kernel_size=config.kernel_size,
        padding='same',
        strides=1,
        use_bias=False,
        kernel_initializer='he_normal'
    )(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    # Primeiro bloco MobileNet, expansion=1 (Linear bottleneck)
    layer = mobilenet_block(layer, config.base_filters, strides=1, config=config, expansion=1)
    
    return layer


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
        
        # Aplicamos o bloco MobileNet com fator de expansão (padrão 2x ou 4x em V2, usamos 2 para ficar ultracompacto)
        layer = mobilenet_block(layer, filters, strides, config, expansion=2)
        
    return layer


def output_block(layer, inputs, rr_input, config):
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = GlobalAveragePooling1D()(layer)
    
    # RR branch
    rr_layer = BatchNormalization()(rr_input)
    rr_layer = Dense(config.rr_units, activation='relu')(rr_layer)
    layer = concatenate([layer, rr_layer])
    
    layer = Dense(config.dense_units, activation='relu')(layer)
    outputs = Dense(len(classes), activation='softmax')(layer)
    
    model = Model(inputs=[inputs, rr_input], outputs=outputs, name='cnn_classifier_mn')

    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def cnn_model(config):
    print('[MODEL] Building MobileNet-1D model...')
    inputs = Input(shape=(config.input_size, 1), name='input')
    rr_input = Input(shape=(len(RR_FEATURES),), name='rr')
    
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    model = output_block(layer, inputs, rr_input, config)
    
    print(f'[MODEL] MobileNet model built and compiled ({model.count_params():,} parameters).')
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, train_inputs):
    print('[EXPORT] Starting model export...')
    output_directory.mkdir(parents=True, exist_ok=True)

    keras_path = output_directory / 'cnn_classifier_mn.keras'
    tflite_path = output_directory / 'cnn_classifier_mn.tflite'
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
            str(output_directory / 'cnn_classifier_mn.keras'),
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
        kernel_size=7,
        n_blocks=6,
        drop_rate=0.2,
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
        trained_model=str(output_directory / 'cnn_classifier_mn.keras'),
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