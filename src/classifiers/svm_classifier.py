from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_data import build_full_dataset, classes
from helpers import extract_neurokit_features


MODEL_PATH = Path("models/svm_classifier.joblib")


def train_svm(x_train, y_train, C_value=0.001, gamma_value=0.0):
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
    return model


def evaluate_model(model, x_test, y_test):
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


def main(C_value=0.001, gamma_value=0.0):
    config = type("SVMConfig", (), {
        "split": True,
        "input_size": 256,
        "feature": "MLII",
    })()
    x_train, y_train, x_test, y_test = build_full_dataset(config)
    x_train, y_train = extract_neurokit_features(x_train, y_train)
    x_test, y_test = extract_neurokit_features(x_test, y_test)

    print("Training SVM...")
    model = train_svm(x_train, y_train, C_value, gamma_value)
    evaluate_model(model, x_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
