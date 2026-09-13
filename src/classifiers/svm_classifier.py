from pathlib import Path
import joblib
import emlearn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from load_data import build_full_dataset, classes
from helpers import extract_neurokit_features


def export_model(model):
    print('[EXPORT] Starting SVM export...')
    output_directory = Path('models')
    output_directory.mkdir(parents=True, exist_ok=True)
    
    c_model = emlearn.convert(model, method="inline")
    c_model.save(file=str(output_directory / "svm_model.h"), name="svm_model")
    print('[EXPORT] Saved SVM joblib and C header models.')


def train_svm(x_train, y_train, C_value=0.001, gamma_value=0.0):
    print('[TRAIN] Starting SVM training...')
    print(f'[TRAIN] Training features shape: {x_train.shape}')
    print(f'[TRAIN] Training labels shape: {y_train.shape}')
    gamma = "auto" if gamma_value == 0.0 else gamma_value
    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=C_value,
            gamma=gamma,
            kernel="rbf",
            class_weight="balanced",
            decision_function_shape="ovo",
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


def main(C_value=0.001, gamma_value=0.0):
    print('[START] SVM classifier execution started.')
    model_path = Path("models/svm_classifier.joblib")
    
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

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    export_model(model)
    print(f"Model saved to {model_path}")
    print('[DONE] SVM classifier execution finished.')


if __name__ == "__main__":
    main()
