#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate a Random Forest using NeuroKit2 ECG features."""

from pathlib import Path
import pickle

import neurokit2 as nk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

from load_data import build_full_dataset, classes

DATASET_PATH = Path("../datasets")
OUTPUT_MODEL_PATH = DATASET_PATH / "heartbeatClassifier.pickle"
SAMPLING_RATE = 150
NUMBER_OF_TREES = 40
BEAT_WINDOW_SECONDS = 0.6

FEATURE_COLUMNS = (
    "ECG_Clean",
    "ECG_Rate",
    "ECG_Quality",
    "ECG_Phase_Atrial",
    "ECG_Phase_Ventricular",
)

OUTPUT_LABELS = [
    BeatType.NORMAL.symbol(),
    BeatType.AURICULAR_PREMATURE_CONTRACTION.symbol(),
    BeatType.PREMATURE_VENTRICULAR_CONTRACTION.symbol(),
    BeatType.FUSION.symbol(),
    BeatType.UNKNOWN.symbol(),
]

def load_signal_dataset(path):
    with path.open("rb") as file:
        return pickle.load(file)

def _numeric_value(values, default=0.0):
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    return float(numeric_values[-1]) if len(numeric_values) else default

def _beat_features(processed_signal, beat_time):
    beat_index = int(round(beat_time * SAMPLING_RATE))
    half_window = int(BEAT_WINDOW_SECONDS * SAMPLING_RATE / 2)
    start = max(0, beat_index - half_window)
    end = min(len(processed_signal), beat_index + half_window)
    window = processed_signal.iloc[start:end]

    features = []
    for column in FEATURE_COLUMNS:
        values = window[column]
        if column == "ECG_Clean":
            numeric_values = pd.to_numeric(values, errors="coerce").dropna()
            if numeric_values.empty:
                features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([
                    float(numeric_values.mean()),
                    float(numeric_values.std()),
                    float(numeric_values.min()),
                    float(numeric_values.max()),
                ])
        else:
            features.append(_numeric_value(values))
    return features

def extract_neurokit_features(dataset):
    feature_rows = []
    labels = []
    sources = []

    for signal, record_labels, record_name in zip(
        dataset["signals"], dataset["labels"], dataset["records"]
    ):
        processed_signal, _ = nk.ecg_process(
            np.asarray(signal),
            sampling_rate=SAMPLING_RATE,
        )
        for label in record_labels:
            beat_type = label["beat"]
            if beat_type == BeatType.OTHER:
                continue
            feature_rows.append(_beat_features(processed_signal, label["time"]))
            labels.append(beat_type.value)
            sources.append(record_name)

    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(sources),
    )

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
    train_dataset = load_signal_dataset(DATASET_PATH / "train_set_signals.pickle")
    test_dataset = load_signal_dataset(DATASET_PATH / "test_set_signals.pickle")

    print("Extracting NeuroKit2 features...")
    train_features, train_labels, train_sources = extract_neurokit_features(train_dataset)
    test_features, test_labels, _ = extract_neurokit_features(test_dataset)

    print("Training model...")
    forest_classifier = RandomForestClassifier(
        random_state=42,
        n_estimators=NUMBER_OF_TREES,
    )
    forest_classifier.fit(train_features, train_labels)
    print(forest_classifier.feature_importances_)

    print("Testing with leave-one-record-out cross-validation...")
    train_predictions = cross_val_predict(
        forest_classifier,
        train_features,
        train_labels,
        cv=leave_one_record_out(train_sources),
    )
    print("Train accuracy:", accuracy_score(train_labels, train_predictions))

    test_predictions = forest_classifier.predict(test_features)
    test_confusion_matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=np.arange(len(OUTPUT_LABELS)),
    )
    print("Test confusion matrix:")
    print(test_confusion_matrix)
    print("Test accuracy:", accuracy_score(test_labels, test_predictions))

    evaluation = evaluate_classifier(test_confusion_matrix, OUTPUT_LABELS)
    print("Evaluation details:")
    print(evaluation)
    print("Classification report:")
    print(
        classification_report(
            test_labels,
            test_predictions,
            labels=np.arange(len(OUTPUT_LABELS)),
            target_names=OUTPUT_LABELS,
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
                "feature_columns": FEATURE_COLUMNS,
            },
            file,
        )
    print(f"Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
