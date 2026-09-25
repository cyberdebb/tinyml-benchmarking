from pathlib import Path
import itertools
import pickle
from types import SimpleNamespace
import emlearn
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import RR_FEATURES, build_full_dataset
from helpers import (extract_neurokit_features, macro_f1, print_aami_report, select_compact_model,
                     smoothed_class_weights)

# Hyperparameter search (helpers.select_compact_model): every combination
# below is trained on the training patients and scored on the validation
# patients. The inlined forest (random_forest_classifier.h) takes flash in
# proportion to its total decision-node count, driven by the number of
# trees, their depth and the leaf size; the search keeps the smallest forest
# among the ones as good as the best. CLASS_WEIGHT_POWERS changes how much
# the rare classes are favored (helpers.smoothed_class_weights): with the
# square root the forest called only 17% of the S beats S on DS2.
RF_N_ESTIMATORS_VALUES = (10, 20, 40)
RF_MAX_DEPTH_VALUES = (8, 12, 16)
RF_MIN_SAMPLES_LEAF_VALUES = (4, 16)
RF_CLASS_WEIGHT_POWERS = (0.5, 0.75, 1.0)

# Estimated flash per decision node once compiled (a float load + compare +
# branch on a 32-bit target). method="inline" turns the forest into nested
# if/else code, so there is no data array whose sizeof() could be reported
# like the other models' -- this is an estimate, not a measurement.
RF_BYTES_PER_NODE = 8


def forest_node_count(forest_classifier):
    return sum(tree.tree_.node_count for tree in forest_classifier.estimators_)


def train_forest(x_train, y_train, n_estimators, max_depth, min_samples_leaf, class_weight_power):
    forest_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        class_weight=smoothed_class_weights(y_train, class_weight_power),
        random_state=42,
        n_jobs=-1,
    )
    return forest_classifier.fit(x_train, y_train)


def search_forest(x_train, y_train, x_validate, y_validate):
    """Trains every combination in the grid and returns the configuration
    chosen by helpers.select_compact_model (a dict of train_forest's
    arguments). The forests are not kept (they can take a lot of memory):
    the chosen one is trained again, with the same seed."""
    results = []
    grid = list(itertools.product(RF_N_ESTIMATORS_VALUES, RF_MAX_DEPTH_VALUES,
                                  RF_MIN_SAMPLES_LEAF_VALUES, RF_CLASS_WEIGHT_POWERS))
    for index, (n_estimators, max_depth, min_samples_leaf, power) in enumerate(grid, start=1):
        forest_classifier = train_forest(x_train, y_train, n_estimators, max_depth,
                                         min_samples_leaf, power)
        score = macro_f1(y_validate, forest_classifier.predict(x_validate))
        results.append({
            'config': {'n_estimators': n_estimators, 'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf, 'class_weight_power': power},
            'val_macro_f1': score,
            'model_bytes': forest_node_count(forest_classifier) * RF_BYTES_PER_NODE,
        })
        print(f'[SEARCH] {index}/{len(grid)} {results[-1]["config"]}: '
              f'validation macro-F1 {score:.4f}, {results[-1]["model_bytes"] / 1024:.1f} KB')
    return select_compact_model('Random Forest', results)['config']


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

    # The generated C source's byte length would be a bad stand-in for the
    # model size (it counts every brace, indent and "return N;", wildly
    # overstating the compiled size). Total decision-node count is a real,
    # reproducible complexity measure; RF_BYTES_PER_NODE converts it into
    # an estimated flash footprint.
    bytes_per_node = RF_BYTES_PER_NODE
    node_count = forest_node_count(forest_classifier)
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
    # The validation patients (held out of DS1) choose the forest's
    # hyperparameters (search_forest); the final metrics are on the test
    # patients (DS2).
    train, validation, test = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {train.X.shape}')
    print(f'[DATA] Validation windows shape: {validation.X.shape}')
    print(f'[DATA] Test windows shape: {test.X.shape}')
    train_features, train_labels = extract_neurokit_features(train.X, train.rr, train.y)
    validation_features, validation_labels = extract_neurokit_features(
        validation.X, validation.rr, validation.y)
    test_features, test_labels = extract_neurokit_features(test.X, test.rr, test.y)

    chosen = search_forest(train_features, train_labels, validation_features, validation_labels)
    print(f'[TRAIN] Training the chosen Random Forest: {chosen}')
    forest_classifier = train_forest(train_features, train_labels, **chosen)
    print('[TRAIN] Random Forest training completed.')
    # 12 morphology features (helpers.beat_features) followed by RR_FEATURES.
    feature_names = [f'morph_{index}' for index in range(12)] + list(RR_FEATURES)
    print('[TRAIN] Feature importances:')
    for name, importance in zip(feature_names, forest_classifier.feature_importances_):
        print(f'  {name}: {importance:.4f}')

    test_predictions = forest_classifier.predict(test_features)
    evaluation = print_aami_report('Random Forest', test_labels, test_predictions, test.records)

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
