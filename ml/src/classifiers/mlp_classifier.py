from types import SimpleNamespace
from pathlib import Path
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score,  precision_score,  recall_score,  f1_score,  roc_curve,  auc,  precision_recall_curve
from sklearn.preprocessing import StandardScaler, label_binarize
import joblib
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset, classes
from helpers import (build_representative_dataset, evaluate_tflite, extract_neurokit_features,
                     smoothed_class_weights)


def train_model_sklearn(x_train, y_train, x_validate, y_validate):
    """
    Train an MLPClassifier model using scikit-learn.

    Parameters:
    - x_train (numpy.ndarray): Training features.
    - y_train (numpy.ndarray): Training labels.

    Returns:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    """
    print('[TRAIN] Starting MLP training...')
    print(f'[TRAIN] Training features shape: {x_train.shape}')
    print(f'[TRAIN] Training labels shape: {y_train.shape}')

    # Create an MLPClassifier
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(256, 64, 16),
        max_iter=1,
        random_state=42,
        warm_start=True,
        early_stopping=False,
    )

    # MLPClassifier has no class_weight, so the same smoothed weights as the
    # other models are applied by resampling: each class ends up with
    # count * weight samples (= sqrt(n_largest * n_class)).
    class_weights = smoothed_class_weights(y_train)
    balanced_indices = []
    random_generator = np.random.default_rng(42)
    for class_id, weight in class_weights.items():
        class_indices = np.flatnonzero(y_train.astype(int) == class_id)
        target_count = int(round(len(class_indices) * weight))
        balanced_indices.extend(
            random_generator.choice(class_indices, size=target_count, replace=True)
        )
    balanced_indices = np.asarray(balanced_indices)

    balanced_x_train = x_train[balanced_indices]
    balanced_y_train = y_train[balanced_indices]
    print(f'[TRAIN] Class weights: {class_weights}')
    print(f'[TRAIN] Resampled training samples: {len(balanced_indices)}')

    best_validation_accuracy = -np.inf
    epochs_without_improvement = 0
    best_coefs = None
    best_intercepts = None
    max_epochs = 400
    early_stopping_patience = 10

    for epoch in range(max_epochs):
        # max_iter=1 + warm_start=True intentionally run one iteration per
        # epoch, so sklearn always warns that it hasn't converged yet; that
        # warning is expected here, not a real problem.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            mlp_classifier.fit(balanced_x_train, balanced_y_train)
        train_accuracy = accuracy_score(
            balanced_y_train,
            mlp_classifier.predict(balanced_x_train),
        )
        validation_accuracy = accuracy_score(
            y_validate,
            mlp_classifier.predict(x_validate),
        )
        print(
            f'[TRAIN] Epoch {epoch + 1}/{max_epochs} - '
            f'loss: {mlp_classifier.loss_:.6f} - '
            f'train_accuracy: {train_accuracy:.4f} - '
            f'validation_accuracy: {validation_accuracy:.4f}'
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            best_coefs = [weights.copy() for weights in mlp_classifier.coefs_]
            best_intercepts = [bias.copy() for bias in mlp_classifier.intercepts_]
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f'[TRAIN] Early stopping at epoch {epoch + 1}. '
                f'Best validation accuracy: {best_validation_accuracy:.4f}'
            )
            break

    # Keep the weights of the best validation epoch, not the last one.
    mlp_classifier.coefs_ = best_coefs
    mlp_classifier.intercepts_ = best_intercepts
    print('[TRAIN] MLP training completed.')

    return mlp_classifier

def evaluate_model(classifier, x_validate, y_validate, directory):
    """
    Evaluate the trained model using scikit-learn metrics.

    Parameters:
    - classifier: Trained classifier model (e.g., MLPClassifier).
    - x_validate (numpy.ndarray): Validation features.
    - y_validate (numpy.ndarray): Validation labels.
    """
    print('[EVALUATION] Starting MLP evaluation...')

    # Make predictions on the validation set
    y_pred = classifier.predict(x_validate)

    # Evaluate accuracy
    accuracy = accuracy_score(y_validate, y_pred)
    print('\nTest Accuracy: {:.2f}%\n'.format(accuracy * 100))

    

def test_model(classifier, x_test, y_test, directory):
    """
    Test the trained model and print additional evaluation metrics.

    Parameters:
    - classifier: Trained classifier model (e.g., MLPClassifier).
    - x_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    """
    print("Testing Phase")

    # Make predictions on the test set
    y_pred = classifier.predict(x_test)

    # Additional evaluation metrics
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print('\nPrecision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))
    

def export_scaler_header(scaler, output_directory):
    """Writes the StandardScaler parameters for the firmware, which
    standardizes the features the same way before quantizing them."""
    scaler_header = output_directory / 'mlp_classifier_scaler.h'
    mean_values = ', '.join(f'{value:.9g}f' for value in scaler.mean_)
    scale_values = ', '.join(f'{value:.9g}f' for value in scaler.scale_)
    scaler_header.write_text(
        '#ifndef MLP_CLASSIFIER_SCALER_H\n'
        '#define MLP_CLASSIFIER_SCALER_H\n\n'
        f'#define MLP_FEATURE_COUNT {len(scaler.mean_)}\n'
        f'static const float mlp_scaler_mean[MLP_FEATURE_COUNT] = {{{mean_values}}};\n'
        f'static const float mlp_scaler_scale[MLP_FEATURE_COUNT] = {{{scale_values}}};\n\n'
        '#endif\n',
        encoding='ascii',
    )
    print(f'[EXPORT] Saved StandardScaler parameters to {scaler_header}.')


def export_model(mlp_classifier, scaler, x_train):
    """
    Export the trained model for serving predictions.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - scaler (StandardScaler): Feature scaler fitted on the training set.
    - x_train (numpy.ndarray): Standardized training features, used only to
      calibrate the int8 quantization ranges (representative_dataset below).

    Returns the path of the .tflite model.
    """
    print('[EXPORT] Starting MLP model export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)

    joblib.dump({'scaler': scaler, 'model': mlp_classifier},
                output_directory / 'mlp_classifier.joblib')
    export_scaler_header(scaler, output_directory)

    keras_model = Sequential([Input(shape=(mlp_classifier.n_features_in_,))])
    for layer_size in mlp_classifier.hidden_layer_sizes:
        keras_model.add(Dense(layer_size, activation='relu'))
    keras_model.add(Dense(len(mlp_classifier.classes_), activation='softmax'))

    keras_model.set_weights(
        [weight for pair in zip(mlp_classifier.coefs_, mlp_classifier.intercepts_) for weight in pair]
    )
    keras_model.save(output_directory / 'mlp_classifier.keras')

    # Full integer (int8) quantization, same approach as cnn_classifier.py's
    # export_model(): weights, activations, input and output all int8,
    # calibrated from the actual training features. Previously this just
    # called converter.convert() with no optimizations at all, so MLP was
    # the only one of the four models shipping fully uncompressed.
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset([x_train])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    tflite_path = output_directory / 'mlp_classifier.tflite'
    tflite_path.write_bytes(tflite_model)
    print(f'[EXPORT] Saved MLP joblib, Keras, and TFLite models '
          f'({len(tflite_model) / 1024:.1f} KB quantized).')
    return tflite_path

def main():
    print('[START] MLP classifier execution started.')
    directory = ''
    config = SimpleNamespace(split=True, input_size=256, feature='MLII')
    print('[DATA] Loading ECG dataset...')
    train, validation, test = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {train.X.shape}')
    print(f'[DATA] Validation windows shape: {validation.X.shape}')
    print(f'[DATA] Test windows shape: {test.X.shape}')
    x_train, y_train = extract_neurokit_features(train.X, train.rr, train.y)
    x_validate, y_validate = extract_neurokit_features(validation.X, validation.rr, validation.y)
    x_test, y_test = extract_neurokit_features(test.X, test.rr, test.y)

    # The features have very different ranges, and the int8 input tensor
    # quantizes all of them with a single scale: without standardizing,
    # small-range features collapse to one or two int8 levels. The firmware
    # applies the same scaler (mlp_classifier_scaler.h) before quantizing.
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_validate = scaler.transform(x_validate).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    trained_mlp_model = train_model_sklearn(
        x_train,
        y_train,
        x_validate,
        y_validate,
    )
    # Final metrics on the test set (DS2): patients never seen in training
    # or in the early stopping done on the validation set.
    evaluate_model(trained_mlp_model, x_test, y_test, directory)
    tflite_path = export_model(trained_mlp_model, scaler, x_train)
    evaluate_tflite(tflite_path, [x_test], y_test)
    print('[DONE] MLP classifier execution finished.')

if __name__ == "__main__":
    main()