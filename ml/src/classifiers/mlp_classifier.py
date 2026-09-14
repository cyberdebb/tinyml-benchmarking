from types import SimpleNamespace
from pathlib import Path
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Input
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score,  precision_score,  recall_score,  f1_score,  roc_curve,  auc,  precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import joblib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.load_data import build_full_dataset, classes
from ml.src.helpers import extract_neurokit_features


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

    class_ids = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=class_ids,
        y=y_train,
    )
    balanced_indices = []
    random_generator = np.random.default_rng(42)
    for class_id in np.unique(y_train):
        class_indices = np.flatnonzero(y_train == class_id)
        target_count = int(np.max(np.bincount(y_train.astype(int))))
        balanced_indices.extend(
            random_generator.choice(class_indices, size=target_count, replace=True)
        )
    balanced_indices = np.asarray(balanced_indices)

    balanced_x_train = x_train[balanced_indices]
    balanced_y_train = y_train[balanced_indices]
    print(f'[TRAIN] Class weights: {dict(zip(class_ids, class_weights))}')
    print(f'[TRAIN] Balanced training samples: {len(balanced_indices)}')

    best_validation_accuracy = -np.inf
    epochs_without_improvement = 0
    best_coefs = None
    best_intercepts = None
    max_epochs = 400
    early_stopping_patience = 10

    for epoch in range(max_epochs):
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

    print('[EVALUATION] Starting MLP validation...')

    # Make predictions on the validation set
    y_pred = classifier.predict(x_validate)

    # Evaluate accuracy
    accuracy = accuracy_score(y_validate, y_pred)
    print('\nTest Accuracy: {:.2f}%\n'.format(accuracy * 100))

    # Confusion Matrix
    cm = confusion_matrix(y_validate, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Precision, Recall, F1 Score
    precision = precision_score(y_validate, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_validate, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_validate, y_pred, average='weighted', zero_division=0)

    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))
    print('[EVALUATION] MLP validation completed.')

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_validate, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{directory}/reports/figures/precision_recall_curve.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_validate, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{directory}/reports/figures/roc_curve.png")

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

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('\nConfusion Matrix:')
    print(cm)

    # Precision, Recall, F1 Score
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print('\nPrecision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1 Score: {:.2f}'.format(f1))

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{directory}/reports/figures/test_precision_recall_curve.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{directory}/reports/figures/test_roc_curve.png")

def export_model(mlp_classifier):
    """
    Export the trained model for serving predictions.

    Parameters:
    - mlp_classifier (MLPClassifier): Trained MLPClassifier model.
    - save_path (str): Path to save the exported model.
    """
    print('[EXPORT] Starting MLP model export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(mlp_classifier, output_directory / 'mlp_classifier.joblib')

    keras_model = Sequential([Input(shape=(mlp_classifier.n_features_in_,))])
    for layer_size in mlp_classifier.hidden_layer_sizes:
        keras_model.add(Dense(layer_size, activation='relu'))
    keras_model.add(Dense(len(mlp_classifier.classes_), activation='softmax'))

    keras_model.set_weights(
        [weight for pair in zip(mlp_classifier.coefs_, mlp_classifier.intercepts_) for weight in pair]
    )
    keras_model.save(output_directory / 'mlp_classifier.keras')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    (output_directory / 'mlp_classifier.tflite').write_bytes(converter.convert())
    print('[EXPORT] Saved MLP joblib, Keras, and TFLite models.')

def main():
    print('[START] MLP classifier execution started.')
    directory = ''
    config = SimpleNamespace(split=True, input_size=256, feature='MLII')
    print('[DATA] Loading ECG dataset...')
    x_train, y_train, x_validate, y_validate = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {x_train.shape}')
    print(f'[DATA] Validation windows shape: {x_validate.shape}')
    x_train, y_train = extract_neurokit_features(x_train, y_train)
    x_validate, y_validate = extract_neurokit_features(x_validate, y_validate)

    trained_mlp_model = train_model_sklearn(
        x_train,
        y_train,
        x_validate,
        y_validate,
    )
    evaluate_model(trained_mlp_model, x_validate, y_validate, directory)
    export_model(trained_mlp_model)
    print('[DONE] MLP classifier execution finished.')

if __name__ == "__main__":
    main()