from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

# Add the source directory so the local module imports below resolve reliably.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset
from helpers import extract_neurokit_features, print_aami_report, smoothed_class_weights


def _format_float(value):
    s = f'{float(value):.9g}'
    if '.' not in s and 'e' not in s.lower():
        s += '.0'
    return s + 'f'


def _write_rows(file, values):
    for row in np.asarray(values):
        file.write('  {' + ','.join(_format_float(value) for value in row) + '},\n')


def _quantization_scale(values, target_max=30000):
    """Largest power-of-friendly int scale that keeps `values * scale`
    inside int16 range, computed from the actual data instead of a fixed
    guess -- so it stays safe if a retrain shifts the value range (e.g. a
    different C or class_weight changes how large dual_coef_ can get).
    target_max (30000, not the full 32767) leaves headroom below the
    int16 ceiling."""
    largest = float(np.abs(values).max())
    if largest == 0.0:
        return 1
    return max(1, int(target_max / largest))


def _write_int16_rows(file, values, scale):
    quantized = np.clip(np.round(np.asarray(values) * scale), -32768, 32767).astype(np.int16)
    for row in quantized:
        file.write('  {' + ','.join(str(int(value)) for value in row) + '},\n')


def export_classifier_header(model, output_directory):
    classifier = model.named_steps['svc']
    support_vectors = classifier.support_vectors_
    support_counts = classifier.n_support_
    support_starts = np.concatenate(([0], np.cumsum(support_counts)[:-1]))
    class_count = len(classifier.classes_)
    pair_count = class_count * (class_count - 1) // 2

    # Support vectors and dual coefficients are stored as int16 fixed-point
    # (roughly halving their flash footprint vs float32) instead of
    # emlearn-style truncation: the scale is derived from this model's own
    # value range (see _quantization_scale), so no precision is thrown away
    # to a scale picked for a different model. The intercept stays float --
    # only 10 values, and its precision directly affects the final sum's
    # sign near the decision boundary.
    sv_scale = _quantization_scale(support_vectors)
    dual_coef_scale = _quantization_scale(classifier.dual_coef_)

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
            f'#define SVM_SV_SCALE {sv_scale}.0f\n'
            f'#define SVM_DUAL_COEF_SCALE {dual_coef_scale}.0f\n'
            'static const int32_t svm_n_support[SVM_CLASS_COUNT] = {'
            + ','.join(str(int(value)) for value in support_counts)
            + '};\n'
            'static const int32_t svm_support_start[SVM_CLASS_COUNT] = {'
            + ','.join(str(int(value)) for value in support_starts)
            + '};\n'
            'static const int16_t svm_support_vectors[SVM_SUPPORT_VECTOR_COUNT][SVM_FEATURE_COUNT] = {\n'
        )
        _write_int16_rows(file, support_vectors, sv_scale)
        file.write('};\n')
        file.write(
            'static const int16_t svm_dual_coef[SVM_CLASS_COUNT - 1][SVM_SUPPORT_VECTOR_COUNT] = {\n'
        )
        _write_int16_rows(file, classifier.dual_coef_, dual_coef_scale)
        file.write('};\n')
        file.write(
            'static const float svm_intercept[SVM_PAIR_COUNT] = {'
            + ','.join(_format_float(value) for value in classifier.intercept_)
            + '};\n'
            # Actual flash footprint of the model's data, computed by the
            # compiler from the arrays above -- so it stays correct even if
            # their sizes change, unlike a value hardcoded at export time.
            '#define SVM_MODEL_BYTES (sizeof(svm_n_support) + sizeof(svm_support_start) '
            '+ sizeof(svm_support_vectors) + sizeof(svm_dual_coef) + sizeof(svm_intercept))\n'
            # One-vs-one decision function, following libsvm's actual
            # svm_predict_values(): the kernel value between x and every
            # support vector is computed once (kvalue[]), and each pairwise
            # classifier (i, j) only sums the two class-specific slices of
            # dual_coef_ -- dual_coef_[j-1] over class i's support vectors
            # and dual_coef_[i] over class j's support vectors. Summing over
            # every support vector with a single dual_coef row (as a naive
            # reading of the arrays suggests) is both wrong (mixes in
            # coefficients that do not apply to this pair) and far slower
            # (SVM_PAIR_COUNT full passes over all support vectors instead
            # of one).
            # kvalue is static (not a local array) because
            # SVM_SUPPORT_VECTOR_COUNT can be in the thousands -- a stack
            # array that size would overflow the ESP32 task stack.
            'static inline int svm_predict(const float *features) {\n'
            '    static float kvalue[SVM_SUPPORT_VECTOR_COUNT];\n'
            '    for (int k = 0; k < SVM_SUPPORT_VECTOR_COUNT; ++k) {\n'
            '        float distance = 0.0f;\n'
            '        for (int f = 0; f < SVM_FEATURE_COUNT; ++f) {\n'
            '            float delta = features[f] - (float)svm_support_vectors[k][f] / SVM_SV_SCALE;\n'
            '            distance += delta * delta;\n'
            '        }\n'
            '        kvalue[k] = expf(-SVM_GAMMA * distance);\n'
            '    }\n'
            '\n'
            '    int votes[SVM_CLASS_COUNT] = {0};\n'
            '    int pair = 0;\n'
            '    for (int i = 0; i < SVM_CLASS_COUNT; ++i) {\n'
            '        const int start_i = svm_support_start[i];\n'
            '        const int count_i = svm_n_support[i];\n'
            '        for (int j = i + 1; j < SVM_CLASS_COUNT; ++j, ++pair) {\n'
            '            const int start_j = svm_support_start[j];\n'
            '            const int count_j = svm_n_support[j];\n'
            '            float sum = svm_intercept[pair];\n'
            '            for (int k = 0; k < count_i; ++k) {\n'
            '                sum += ((float)svm_dual_coef[j - 1][start_i + k] / SVM_DUAL_COEF_SCALE) * kvalue[start_i + k];\n'
            '            }\n'
            '            for (int k = 0; k < count_j; ++k) {\n'
            '                sum += ((float)svm_dual_coef[i][start_j + k] / SVM_DUAL_COEF_SCALE) * kvalue[start_j + k];\n'
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
    print(f'[EXPORT] Saved SVM classifier C header to {classifier_header} '
          f'(sv_scale={sv_scale}, dual_coef_scale={dual_coef_scale}).')


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
    # Same class weights as the other models (helpers.smoothed_class_weights).
    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=C_value,
            gamma=gamma,
            kernel="rbf",
            class_weight=smoothed_class_weights(y_train),
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
    print_aami_report('SVM', y_test, predictions)
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
    # The validation set (patients held out of DS1) is only used by the
    # models with early stopping (CNN, MLP); SVM trains on the same training
    # patients and is evaluated on the same test patients (DS2).
    train, _, test = build_full_dataset(config)
    print(f'[DATA] Training windows shape: {train.X.shape}')
    print(f'[DATA] Test windows shape: {test.X.shape}')
    x_train, y_train = extract_neurokit_features(train.X, train.rr, train.y)
    x_test, y_test = extract_neurokit_features(test.X, test.rr, test.y)

    print("Training SVM...")
    model = train_svm(x_train, y_train, C_value, gamma_value)
    evaluate_model(model, x_test, y_test)

    joblib.dump(model, model_path)
    export_model(model)
    print(f"Model saved to {model_path}")
    print('[DONE] SVM classifier execution finished.')


if __name__ == "__main__":
    main()
