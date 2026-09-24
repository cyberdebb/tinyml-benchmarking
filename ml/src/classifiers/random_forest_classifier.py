from pathlib import Path
import pickle
from types import SimpleNamespace
import emlearn
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import RR_FEATURES, build_full_dataset
from helpers import extract_neurokit_features, print_aami_report, smoothed_class_weights

# Size/accuracy tradeoff for the embedded target: RANDOM_FOREST_MODEL_BYTES
# (random_forest_classifier.h) scales directly with the forest's total
# decision-node count, which is driven by these three. The previous values
# (40, 20, 2) produced ~150k nodes and an estimated ~1.36MB inlined model.
# Fewer/shallower trees with larger leaves cut that a lot, but there's no
# dataset here to measure the resulting accuracy -- re-run the on-device
# benchmark after retraining with these values and compare against the
# previous run before trusting them for a final result. Tune further from
# here if that comparison isn't the tradeoff you want.
RF_N_ESTIMATORS = 20       # was 40 -- roughly halves node count linearly
RF_MAX_DEPTH = 12          # was 20 -- cuts node count per tree the most
RF_MIN_SAMPLES_LEAF = 4    # was 2 -- fewer, larger leaves


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

def main():
    print('[START] Random Forest classifier execution started.')
    config = SimpleNamespace(split=True, input_size=256, feature='MLII')
    print(f'[CONFIG] Configuration: {vars(config)}')
    print('[DATA] Loading ECG dataset...')
    # The validation set (patients held out of DS1) is only used by the
    # models with early stopping (CNN, MLP); RF trains on the same training
    # patients and is evaluated on the same test patients (DS2).
    train, _, test = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {train.X.shape}')
    print(f'[DATA] Test windows shape: {test.X.shape}')
    train_features, train_labels = extract_neurokit_features(train.X, train.rr, train.y)
    test_features, test_labels = extract_neurokit_features(test.X, test.rr, test.y)

    print("Training model...")
    forest_classifier = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features='sqrt',
        class_weight=smoothed_class_weights(train_labels),
        random_state=42,
        verbose=1,
    )
    forest_classifier.fit(train_features, train_labels)
    print('[TRAIN] Random Forest training completed.')
    # 12 morphology features (helpers.beat_features) followed by RR_FEATURES.
    feature_names = [f'morph_{index}' for index in range(12)] + list(RR_FEATURES)
    print('[TRAIN] Feature importances:')
    for name, importance in zip(feature_names, forest_classifier.feature_importances_):
        print(f'  {name}: {importance:.4f}')

    test_predictions = forest_classifier.predict(test_features)
    evaluation = print_aami_report('Random Forest', test_labels, test_predictions)

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
