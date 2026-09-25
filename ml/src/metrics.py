"""Test metrics shared by the training scripts (helpers.py) and the board
benchmark (benchmark_serial.py). Only numpy/pandas/scikit-learn, so the
benchmark can use it without TensorFlow."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from load_data import SCORED_CLASSES, classes

SCORED_CLASS_IDS = [classes.index(name) for name in SCORED_CLASSES]


def macro_f1(y_true, y_pred):
    """Mean F1 over SCORED_CLASSES (N, S, V, F). The metric the models are
    compared on and the one used to pick the best epoch/checkpoint: accuracy
    is dominated by N (always answering N already scores ~0.89 on DS2)."""
    return f1_score(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                    labels=SCORED_CLASS_IDS, average='macro', zero_division=0)


def per_record_report(y_true, y_pred, records):
    """One row per record (patient): its beats, and for S, V and F how many
    it has, the sensitivity on them and how many of its beats of other
    classes were called that class (false positives). In the inter-patient
    split a class often comes mostly from one or two patients, and this
    shows which ones decide the class metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    records = np.asarray(records).astype(int)
    rows = []
    for record in np.unique(records):
        in_record = records == record
        row = {'Record': record, 'Beats': int(in_record.sum())}
        for name in ('S', 'V', 'F'):
            label = classes.index(name)
            is_class = in_record & (y_true == label)
            row[name] = int(is_class.sum())
            row[f'{name} Se'] = (np.mean(y_pred[is_class] == label)
                                 if is_class.any() else float('nan'))
            row[f'{name} FP'] = int(np.sum(in_record & (y_true != label) & (y_pred == label)))
        rows.append(row)
    return pd.DataFrame(rows).set_index('Record')


def aami_report(name, y_true, y_pred, records=None):
    """Returns (text, DataFrame) with the test metrics every classifier
    reports; the DataFrame has one row per class. With records (the
    MIT-BIH record of each beat), the text also has per_record_report().

    Per class, one-vs-rest from the confusion matrix (the AAMI EC57 / de
    Chazal metrics):
      Se  (sensitivity, recall)  = TP / (TP + FN)
      +P  (positive predictivity) = TP / (TP + FP)
      FPR (false positive rate)   = FP / (FP + TN)
      F1                          = 2TP / (2TP + FP + FN)
    plus the accuracy, the accuracy of always answering N (the baseline the
    accuracy has to beat) and the macro-F1 over SCORED_CLASSES.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    total = matrix.sum()

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float('nan')

    rows = {'Se': [], '+P': [], 'FPR': [], 'F1': [], 'Support': []}
    for index in range(len(classes)):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        rows['Se'].append(ratio(true_positive, true_positive + false_negative))
        rows['+P'].append(ratio(true_positive, true_positive + false_positive))
        rows['FPR'].append(ratio(false_positive, false_positive + true_negative))
        rows['F1'].append(ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative))
        rows['Support'].append(int(matrix[index, :].sum()))
    report = pd.DataFrame(rows, index=classes)

    accuracy = accuracy_score(y_true, y_pred)
    always_normal = np.mean(y_true == classes.index('N'))
    text = '\n'.join([
        f'[EVALUATION] {name} -- test set ({len(y_true)} beats)',
        f'Confusion matrix (rows = true, columns = predicted: {", ".join(classes)}):',
        str(matrix),
        report.to_string(float_format='{:.4f}'.format),
        f'Accuracy: {accuracy:.4f} (always predicting N: {always_normal:.4f})',
        f'Macro-F1 ({", ".join(SCORED_CLASSES)}): {macro_f1(y_true, y_pred):.4f}',
    ])
    if records is not None:
        by_record = per_record_report(y_true, y_pred, records)
        text += ('\nPer record (Se = sensitivity on the record\'s beats of that class, '
                 'FP = its beats of other classes called that class):\n'
                 + by_record.to_string(float_format='{:.3f}'.format, na_rep='-'))
    return text, report


def print_aami_report(name, y_true, y_pred, records=None):
    """Prints aami_report() and returns its DataFrame."""
    text, report = aami_report(name, y_true, y_pred, records)
    print(text)
    return report
