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


def export_model(model):
    print('[EXPORT] Starting SVM export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)

    scaler = model.named_steps['standardscaler']
    scaler_header = output_directory / 'svm_scaler.h'
    mean_values = ', '.join(f'{value:.9g}f' for value in scaler.mean_)
    scale_values = ', '.join(f'{value:.9g}f' for value in scaler.scale_)
    scaler_header.write_text(
        '#ifndef SVM_SCALER_H\n'
        '#define SVM_SCALER_H\n\n'
        f'#define SVM_FEATURE_COUNT {len(scaler.mean_)}\n'
        f'static const float svm_scaler_mean[SVM_FEATURE_COUNT] = {{{mean_values}}};\n'
        f'static const float svm_scaler_scale[SVM_FEATURE_COUNT] = {{{scale_values}}};\n\n'
        '#endif\n',
        encoding='ascii',
    )
    print(f'[EXPORT] Saved SVM pipeline to models/svm_classifier.joblib.')
    print(f'[EXPORT] Saved StandardScaler parameters to {scaler_header}.')
    print('[EXPORT] emlearn does not support sklearn SVC; no SVM classifier C header was generated.')


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
