from pathlib import Path
import pickle
from types import SimpleNamespace
import emlearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset, classes
from helpers import extract_neurokit_features


def export_model(forest_classifier):
    print('[EXPORT] Starting Random Forest export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)

    # dtype='float' avoids emlearn's default int16 quantization, which
    # truncates each split threshold to int(value) with no scaling -- for
    # small-magnitude ECG statistics (means, stds well under 1.0) that
    # collapses almost every threshold to -1/0/1/2, destroying the tree.
    c_model = emlearn.convert(forest_classifier, method="inline", dtype="float")
    code = c_model.save(name="random_forest")

    # method="inline" turns the forest into nested if/else code, not a data
    # array, so there is no sizeof()-able "model" to report like the TFLite
    # models' embedded .tflite blob or the SVM's parameter arrays -- and the
    # generated C source's byte length is a bad stand-in (it counts every
    # brace, indent and "return N;", wildly overstating the compiled size).
    # Total decision-node count across the forest is a real, reproducible
    # complexity measure; RANDOM_FOREST_BYTES_PER_NODE converts it into an
    # estimated flash footprint (a compiled node is a float load + compare +
    # branch, roughly that many bytes on a 32-bit target) -- an estimate,
    # not a measurement, unlike the other models' reported sizes.
    bytes_per_node = 8
    node_count = sum(tree.tree_.node_count for tree in forest_classifier.estimators_)
    model_bytes = node_count * bytes_per_node

    header_path = output_directory / "random_forest_classifier.h"
    with header_path.open('w', encoding='ascii', newline='\n') as file:
        file.write(f'#define RANDOM_FOREST_NODE_COUNT {node_count}\n')
        file.write(f'#define RANDOM_FOREST_MODEL_BYTES {model_bytes}  '
                    f'/* estimated: node count * {bytes_per_node} bytes/node */\n')
        file.write(code)
    print(f'[EXPORT] Forest has {node_count} decision nodes '
          f'(~{model_bytes / 1024:.1f} KB estimated).')
    print('[EXPORT] Saved Random Forest C header model.')

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
    print('[START] Random Forest classifier execution started.')
    config = SimpleNamespace(split=True, input_size=256, feature='MLII')
    print(f'[CONFIG] Configuration: {vars(config)}')
    print('[DATA] Loading ECG dataset...')
    train_features, train_labels, test_features, test_labels = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {train_features.shape}')
    print(f'[DATA] Test windows shape: {test_features.shape}')
    train_features, train_labels = extract_neurokit_features(train_features, train_labels)
    test_features, test_labels = extract_neurokit_features(test_features, test_labels)

    print("Training model...")
    forest_classifier = RandomForestClassifier(
        n_estimators=40,
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        verbose=1,
    )
    forest_classifier.fit(train_features, train_labels)
    print('[TRAIN] Random Forest training completed.')
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
    
    model_path = Path("models/random_forest_classifier.pickle")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(
            {
                "preprocessor": None,
                "model": forest_classifier,
                "Evaluation_test": evaluation,
            },
            file,
        )
    export_model(forest_classifier)
    print(f"Model saved to {model_path}")
    print('[DONE] Random Forest classifier execution finished.')

if __name__ == "__main__":
    main()
