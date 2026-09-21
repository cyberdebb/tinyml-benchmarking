from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

# Add the source directory so the local module imports below resolve reliably.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset, classes
from helpers import extract_neurokit_features


def _format_float(value):
    s = f'{float(value):.9g}'
    if '.' not in s and 'e' not in s.lower():
        s += '.0'
    return s + 'f'


def _write_rows(file, values):
    for row in np.asarray(values):
        file.write('  {' + ','.join(_format_float(value) for value in row) + '},\n')


def export_classifier_header(model, output_directory):
    classifier = model.named_steps['svc']
    support_vectors = classifier.support_vectors_
    support_counts = classifier.n_support_
    support_starts = np.concatenate(([0], np.cumsum(support_counts)[:-1]))
    class_count = len(classifier.classes_)
    pair_count = class_count * (class_count - 1) // 2

    classifier_header = output_directory / 'svm_classifier.h'
    with classifier_header.open('w', encoding='ascii', newline='\n') as file:
        file.write(
            '#ifndef SVM_CLASSIFIER_H\n'
            '#define SVM_CLASSIFIER_H\n'
            '#include <math.h>\n'
            '#include <stdint.h>\n'
            f'#define SVM_CLASS_COUNT {class_count}\n'
            f'#define SVM_FEATURE_COUNT {support_vectors.shape[1]}\n'
            f'#define SVM_SUPPORT_VECTOR_COUNT {len(support_vectors)}\n'
            f'#define SVM_PAIR_COUNT {pair_count}\n'
            f'#define SVM_GAMMA {_format_float(classifier._gamma)}\n'
            'static const int32_t svm_n_support[SVM_CLASS_COUNT] = {'
            + ','.join(str(int(value)) for value in support_counts)
            + '};\n'
            'static const int32_t svm_support_start[SVM_CLASS_COUNT] = {'
            + ','.join(str(int(value)) for value in support_starts)
            + '};\n'
            'static const float svm_support_vectors[SVM_SUPPORT_VECTOR_COUNT][SVM_FEATURE_COUNT] = {\n'
        )
        _write_rows(file, support_vectors)
        file.write('};\n')
        file.write(
            'static const float svm_dual_coef[SVM_CLASS_COUNT - 1][SVM_SUPPORT_VECTOR_COUNT] = {\n'
        )
        _write_rows(file, classifier.dual_coef_)
        file.write('};\n')
        file.write(
            'static const float svm_intercept[SVM_PAIR_COUNT] = {'
            + ','.join(_format_float(value) for value in classifier.intercept_)
            + '};\n'
            'static inline int svm_predict(const float *features) {\n'
            '    int votes[SVM_CLASS_COUNT] = {0};\n'
            '    int pair = 0;\n'
            '    for (int i = 0; i < SVM_CLASS_COUNT; ++i) {\n'
            '        for (int j = i + 1; j < SVM_CLASS_COUNT; ++j, ++pair) {\n'
            '            float sum = svm_intercept[pair];\n'
            '            for (int k = 0; k < SVM_SUPPORT_VECTOR_COUNT; ++k) {\n'
            '                float distance = 0.0f;\n'
            '                for (int f = 0; f < SVM_FEATURE_COUNT; ++f) {\n'
            '                    float delta = features[f] - svm_support_vectors[k][f];\n'
            '                    distance += delta * delta;\n'
            '                }\n'
            '                sum += svm_dual_coef[j - 1][k] * expf(-SVM_GAMMA * distance);\n'
            '            }\n'
            '            if (sum > 0.0f) ++votes[i]; else ++votes[j];\n'
            '        }\n'
            '    }\n'
            '    int best = 0;\n'
            '    for (int i = 1; i < SVM_CLASS_COUNT; ++i) {\n'
            '        if (votes[i] > votes[best]) best = i;\n'
            '    }\n'
            '    return best;\n'
            '}\n'
            '#endif\n'
        )
    print(f'[EXPORT] Saved SVM classifier C header to {classifier_header}.')


def export_model(model):
    print('[EXPORT] Starting SVM export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)

    scaler = model.named_steps['standardscaler']
    scaler_header = output_directory / 'svm_classifier_scaler.h'
    mean_values = ', '.join(f'{value:.9g}f' for value in scaler.mean_)
    scale_values = ', '.join(f'{value:.9g}f' for value in scaler.scale_)
    scaler_header.write_text(
        '#ifndef SVM_CLASSIFIER_SCALER_H\n'
        '#define SVM_CLASSIFIER_SCALER_H\n\n'
        f'#ifndef SVM_FEATURE_COUNT\n'
        f'#define SVM_FEATURE_COUNT {len(scaler.mean_)}\n'
        f'#endif\n'
        f'static const float svm_scaler_mean[SVM_FEATURE_COUNT] = {{{mean_values}}};\n'
        f'static const float svm_scaler_scale[SVM_FEATURE_COUNT] = {{{scale_values}}};\n\n'
        '#endif\n',
        encoding='ascii',
    )
    export_classifier_header(model, output_directory)
    print(f'[EXPORT] Saved StandardScaler parameters to {scaler_header}.')


def train_svm(x_train, y_train, C_value=1.0, gamma_value=0.0):
    print('[TRAIN] Starting SVM training...')
    print(f'[TRAIN] Training features shape: {x_train.shape}')
    print(f'[TRAIN] Training labels shape: {y_train.shape}')
    gamma = "scale" if gamma_value == 0.0 else gamma_value
    class_ids, class_counts = np.unique(y_train, return_counts=True)
    maximum_count = np.max(class_counts)
    class_weight = {
        class_id: float(np.sqrt(maximum_count / count))
        for class_id, count in zip(class_ids, class_counts)
    }
    print(f'[TRAIN] Smoothed class weights: {class_weight}')
    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=C_value,
            gamma=gamma,
            kernel="rbf",
            class_weight=class_weight,
            decision_function_shape="ovo",
            probability=False,
            max_iter=10000,
            verbose=True,
        ),
    )
    model.fit(x_train, y_train)
    print('[TRAIN] SVM training completed.')
    return model


def evaluate_model(model, x_test, y_test):
    print('[EVALUATION] Starting SVM evaluation...')
    predictions = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=np.arange(len(classes))))
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0,
        )
    )
    print('[EVALUATION] SVM evaluation completed.')


def main(C_value=1.0, gamma_value=0.0):
    print('[START] SVM classifier execution started.')

    model_path = Path("models/svm_classifier.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = type("SVMConfig", (), {
        "split": True,
        "input_size": 256,
        "feature": "MLII",
    })()
    print(f'[CONFIG] C={C_value}, gamma={gamma_value}')
    print('[DATA] Loading ECG dataset...')
    x_train, y_train, x_test, y_test = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {x_train.shape}')
    print(f'[DATA] Test windows shape: {x_test.shape}')
    x_train, y_train = extract_neurokit_features(x_train, y_train)
    x_test, y_test = extract_neurokit_features(x_test, y_test)

    print("Training SVM...")
    model = train_svm(x_train, y_train, C_value, gamma_value)
    evaluate_model(model, x_test, y_test)

    joblib.dump(model, model_path)
    export_model(model)
    print(f"Model saved to {model_path}")
    print('[DONE] SVM classifier execution finished.')


if __name__ == "__main__":
    main()
