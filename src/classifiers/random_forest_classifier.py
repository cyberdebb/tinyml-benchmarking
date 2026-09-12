from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

from load_data import build_full_dataset, classes
from helpers import extract_neurokit_features


OUTPUT_MODEL_PATH = Path("models/heartbeatClassifier.pickle")

def leave_one_record_out(sources):
    for source in np.unique(sources):
        test_indexes = np.flatnonzero(sources == source)
        train_indexes = np.flatnonzero(sources != source)
        yield train_indexes, test_indexes

def evaluate_classifier(confusion_matrix_values, outputs):
    quality_measures = ["Se", "Sp", "Pp", "FPR", "Ac", "F1"]
    quality = np.empty((len(quality_measures), len(outputs)))

    for index, _ in enumerate(outputs):
        true_positive = confusion_matrix_values[index, index]
        false_negative = np.sum(confusion_matrix_values[index, :]) - true_positive
        true_negative = np.sum(confusion_matrix_values) - np.sum(confusion_matrix_values[index, :])
        false_positive = np.sum(confusion_matrix_values[:, index]) - true_positive

        sensitivity_denominator = true_positive + false_negative
        specificity_denominator = true_negative + false_positive
        precision_denominator = true_positive + false_positive
        f1_denominator = 2 * true_positive + false_positive + false_negative

        quality[0, index] = true_positive / sensitivity_denominator
        quality[1, index] = true_negative / specificity_denominator
        quality[2, index] = true_positive / precision_denominator
        quality[3, index] = false_positive / sensitivity_denominator
        quality[4, index] = (true_positive + true_negative) / np.sum(confusion_matrix_values)
        quality[5, index] = 2 * true_positive / f1_denominator

    return pd.DataFrame(quality, columns=outputs, index=quality_measures)

def main():
    config = SimpleNamespace(split=True, input_size=256, feature='MLII')
    train_features, train_labels, test_features, test_labels = build_full_dataset(config)
    train_features, train_labels = extract_neurokit_features(train_features, train_labels)
    test_features, test_labels = extract_neurokit_features(test_features, test_labels)

    print("Training model...")
    forest_classifier = RandomForestClassifier(
        random_state=42,
        n_estimators=40,
    )
    forest_classifier.fit(train_features, train_labels)
    print(forest_classifier.feature_importances_)

    test_predictions = forest_classifier.predict(test_features)
    test_confusion_matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=np.arange(len(classes)),
    )
    print("Test confusion matrix:")
    print(test_confusion_matrix)
    print("Test accuracy:", accuracy_score(test_labels, test_predictions))

    evaluation = evaluate_classifier(test_confusion_matrix, classes)
    print("Evaluation details:")
    print(evaluation)
    print("Classification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    with OUTPUT_MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "preprocessor": None,
                "model": forest_classifier,
                "Evaluation_test": evaluation,
            },
            file,
        )
    print(f"Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
