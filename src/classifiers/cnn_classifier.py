import numpy as np
from types import SimpleNamespace
import tensorflow as tf

from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling1D, add
from keras.models import Model
from keras.optimizers import Adam
from keras.saving import register_keras_serializable

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from load_data import build_full_dataset, classes
from helpers import print_results, mkdir_recursive

@register_keras_serializable(package="custom")
def zeropad(x):
    y = tf.zeros_like(x)
    return tf.concat([x, y], axis=2)

@register_keras_serializable(package="custom")
def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)



def cnn_get_config():
    return SimpleNamespace(
        split=True,
        input_size=256,
        filter_length=32,
        kernel_size=16,
        drop_rate=0.2,
        feature='MLII',
        epochs=80,
        batch=256,
        patience=10,
        min_lr=0.00005,
        checkpoint_path=None,
        resume_epoch=0,
    )

def first_conv_block(inputs, config):
    layer = Conv1D(
        filters=config.filter_length,
        kernel_size=config.kernel_size,
        padding='same',
        strides=1,
        kernel_initializer='he_normal',
    )(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    shortcut = MaxPooling1D(pool_size=1, strides=1)(layer)

    layer = Conv1D(
        filters=config.filter_length,
        kernel_size=config.kernel_size,
        padding='same',
        strides=1,
        kernel_initializer='he_normal',
    )(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(config.drop_rate)(layer)
    layer = Conv1D(
        filters=config.filter_length,
        kernel_size=config.kernel_size,
        padding='same',
        strides=1,
        kernel_initializer='he_normal',
    )(layer)
    return add([shortcut, layer])


def main_loop_blocks(layer, config):
    filter_length = config.filter_length
    n_blocks = 15

    for block_index in range(n_blocks):
        subsample_length = 2 if block_index % 2 == 0 else 1
        shortcut = MaxPooling1D(pool_size=subsample_length)(layer)

        if block_index % 4 == 0 and block_index > 0:
            shortcut = Lambda(
                zeropad,
                output_shape=zeropad_output_shape,
            )(shortcut)
            filter_length *= 2

        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv1D(
            filters=filter_length,
            kernel_size=config.kernel_size,
            padding='same',
            strides=subsample_length,
            kernel_initializer='he_normal',
        )(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(config.drop_rate)(layer)
        layer = Conv1D(
            filters=filter_length,
            kernel_size=config.kernel_size,
            padding='same',
            strides=1,
            kernel_initializer='he_normal',
        )(layer)
        layer = add([shortcut, layer])

    return layer


def output_block(layer, inputs, config):
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Flatten()(layer)
    outputs = Dense(len(classes), activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
    )
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


def cnn_model(config):
    inputs = Input(shape=(config.input_size, 1), name='input')
    layer = first_conv_block(inputs, config)
    layer = main_loop_blocks(layer, config)
    return output_block(layer, inputs, config)


def convert_keras_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path = Path(output_path)
    mkdir_recursive(str(output_path.parent))
    output_path.write_bytes(tflite_model)
    print(f"TFLite model saved to {output_path}")


def prepare_training_data(config, X, y, Xval, yval):
    Xe = np.expand_dims(X, axis=2)
    if not config.split:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            Xe,
            y,
            test_size=0.2,
            random_state=1,
        )

    Xvale = np.expand_dims(Xval, axis=2)
    print("Data shapes before training - Xe:", Xe.shape, "y:", y.shape)
    print("Val shapes before training - Xvale:", Xvale.shape, "yval:", yval.shape)
    print("Final shapes - Xe:", Xe.shape, "y:", y.shape)
    print("Final val shapes - Xvale:", Xvale.shape, "yval:", yval.shape)
    return Xe, Xvale, y, yval


def build_training_callbacks(config):
    return [
        EarlyStopping(patience=config.patience, verbose=1),
        ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            min_lr=config.min_lr,
            verbose=1,
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        ),
        ModelCheckpoint(
            'models/{}-latest.keras'.format(config.feature),
            monitor='val_loss',
            save_best_only=False,
            verbose=1,
            save_freq=10,
        ),
    ]


def cnn_train(config, X, y, Xval=None, yval=None):
    print("Initial shapes - X:", X.shape, "y:", y.shape)
    print("Initial validation shapes - Xval:", Xval.shape if Xval is not None else None, "yval:", yval.shape if yval is not None else None)
    print("Any NaN in initial X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))

    Xe, Xvale, y, yval = prepare_training_data(config, X, y, Xval, yval)

    if config.checkpoint_path is not None:
        model = models.cnn_model(config.checkpoint_path)
        initial_epoch = config.resume_epoch
    else:
        model = cnn_model(config)
        initial_epoch = 0

    mkdir_recursive('models')

    if np.any(np.isnan(Xe)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains None/NaN values")
    if np.any(np.isnan(Xvale)) or np.any(np.isnan(yval)):
        raise ValueError("Validation data contains None/NaN values")

    model.fit(
        Xe,
        y,
        validation_data=(Xvale, yval),
        epochs=config.epochs,
        batch_size=config.batch,
        callbacks=build_training_callbacks(config),
        initial_epoch=initial_epoch,
    )
    convert_keras_to_tflite(model, f'models/{config.feature}-latest.tflite')
    print_results(config, model, Xvale, yval, classes)

def main(config):
    X, y, Xval, yval = build_full_dataset(config)
    cnn_train(config, X, y, Xval, yval)

if __name__ == "__main__":
    config = cnn_get_config()
    main(config)