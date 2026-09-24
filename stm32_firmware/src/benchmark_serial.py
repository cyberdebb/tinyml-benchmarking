"""Sends ECG beats to the ESP32 over UART and collects the predictions.

Usage:
    python src/benchmark_serial.py COM5 <model> --beats 500

The firmware answers every beat with a line:
    RESULT,<predicted_class>,<inference_microseconds>

Results are written to results/<model>_serial.csv so the four models can be
compared later.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import serial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, fbeta_score, confusion_matrix

dataset_directory = Path(__file__).resolve().parents[2] / 'ml'
firmware_directory = Path(__file__).resolve().parents[1]
results_directory = firmware_directory / 'results'

# The test split and the cache location come from the training code itself,
# so the boards are always benchmarked on the same patients the models were
# tested on.
sys.path.insert(0, str(dataset_directory / 'src'))
from load_data import RR_FEATURES, TEST_RECORDS, dataset_cache_path, select_records  # noqa: E402

# Must match the classifier configs and the pipeline dataset_config.
dataset_config = SimpleNamespace(feature='MLII', input_size=256)

baud_rate = 115200
# One beat takes a few hundred ms to travel at 115200 baud, so the timeout has
# to cover transfer plus inference.
read_timeout_seconds = 15
class_names = ('N', 'S', 'V', 'F', 'Q')


def load_test_set():
    """Loads the test split (DS2 patients) every classifier was evaluated on.

    The cache holds the raw, unfiltered ECG windows (see ml/src/load_data.py):
    the band-pass filter is applied only at training/evaluation time on the
    host, never saved back to the cache. So the beats sent here over serial
    are raw, matching what the firmware itself filters on-device
    (tinyml_app_<model>.cc, filter_sos) before running inference.
    """
    cache_file = dataset_cache_path(dataset_config.feature, dataset_config.input_size)
    if not cache_file.exists():
        print(f'[ERROR] Dataset cache not found: {cache_file}')
        print('[ERROR] Run the training (pipeline step 1) first.')
        sys.exit(1)

    with np.load(cache_file) as data:
        if 'rr_features' not in data.files or tuple(data['rr_features']) != RR_FEATURES:
            print('[ERROR] Dataset cache is from an older version (other RR features).')
            print('[ERROR] Run the training (pipeline step 1) again to rebuild it.')
            sys.exit(1)
        X_total = data['X']
        rr_total = data['rr']
        y_total = data['y']
        records = data['records']

    mask = select_records(records, TEST_RECORDS)
    print(f'[DATA] Test set: {mask.sum()} beats')
    return X_total[mask], rr_total[mask], y_total[mask].astype(int)


def select_beats(Xtest, rrtest, ytest, n_beats, seed=1):
    """Picks a random subset, keeping the class distribution of the full set."""
    if n_beats >= len(Xtest):
        return Xtest, rrtest, ytest
    indices = np.arange(len(Xtest))
    try:
        indices, _ = train_test_split(
            indices, train_size=n_beats, stratify=ytest, random_state=seed)
    except ValueError:
        # A class too rare to stratify (fewer than 2 beats): plain random subset.
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(Xtest), size=n_beats, replace=False)
    return Xtest[indices], rrtest[indices], ytest[indices]


def wait_for_ready(connection, timeout=20):
    """Reads the boot log until the firmware reports it is ready."""
    deadline = time.time() + timeout
    model_info = {}
    while time.time() < deadline:
        line = connection.readline().decode('utf-8', errors='replace').strip()
        if not line:
            continue
        print(f'  [stm32] {line}')
        if line.startswith('INFO,'):
            parts = line.split(',')
            model_info = {
                'model': parts[1],
                'model_bytes': int(parts[2]),
                'arena_bytes': int(parts[3]),
            }
        if 'Ready.' in line:
            return model_info
    print('[ERROR] Timed out waiting for the firmware to become ready.')
    print('[ERROR] Try pressing the board reset button, or check the COM port.')
    sys.exit(1)


def send_beat(connection, beat, rr):
    """Sends one beat followed by its RR intervals (load_data.RR_FEATURES)
    and returns (predicted_class, inference_microseconds, filter_microseconds)."""
    payload = ','.join(f'{value:.5f}' for value in (*beat, *rr)) + '\n'
    connection.reset_input_buffer()
    connection.write(payload.encode('ascii'))
    connection.flush()

    deadline = time.time() + read_timeout_seconds
    while time.time() < deadline:
        line = connection.readline().decode('utf-8', errors='replace').strip()
        if not line:
            continue
        if line.startswith('RESULT,'):
            _, prediction, inference_us, filter_us = line.split(',')
            return int(prediction), int(inference_us), int(filter_us)
        print(f'  [stm32] {line}')
    raise TimeoutError('No RESULT line received from the firmware.')


def print_summary(model, y_true, y_predicted, inference_us, filter_us, model_info):
    inference_ms = np.array(inference_us) / 1000.0
    filter_ms = np.array(filter_us) / 1000.0

    # 1. Acurácia
    accuracy = accuracy_score(y_true, y_predicted)

    # Para problemas multiclasse, usamos 'macro' para fazer a média aritmética das métricas de todas as classes
    avg_method = 'macro'

    # 2. Sensibilidade (Recall)
    sensitivity = recall_score(y_true, y_predicted, average=avg_method, zero_division=0)

    # 3. F1-Score
    f1 = f1_score(y_true, y_predicted, average=avg_method, zero_division=0)

    # 4. F-beta Score (usando beta=2.0 como exemplo para dar mais peso à sensibilidade)
    beta_val = 2.0
    f_beta = fbeta_score(y_true, y_predicted, beta=beta_val, average=avg_method, zero_division=0)

    # 5. Especificidade
    # A especificidade em multiclasse precisa ser calculada manualmente extraindo os Verdadeiros Negativos (TN)
    # e Falsos Positivos (FP) da Matriz de Confusão para cada classe.
    # Fixed labels so the matrix is always 5x5, even when a rare class is
    # missing from the beats sent.
    cm = confusion_matrix(y_true, y_predicted, labels=np.arange(len(class_names)))
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        # Evita divisão por zero
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    
    # Média das especificidades de todas as classes
    specificity = np.mean(specificities)
    
    # 6. Vazão Computacional (Throughput)
    # Convertendo o tempo médio de inferência para segundos e dividindo 1 por ele
    mean_inference_sec = np.mean(inference_us) / 1_000_000.0
    throughput = 1.0 / mean_inference_sec if mean_inference_sec > 0 else 0.0

    print('\n==================================================')
    print(f'  Model: {model}')
    if model_info:
        print(f"  Model size: {model_info['model_bytes'] / 1024:.1f} KB")
        print(f"  Arena used: {model_info['arena_bytes'] / 1024:.1f} KB")
    print(f'  Beats: {len(y_true)}')
    print(f'  Accuracy:      {accuracy:.4f}')
    print(f'  Sensitivity:   {sensitivity:.4f} (Macro Avg)')
    print(f'  Specificity:   {specificity:.4f} (Macro Avg)')
    print(f'  F1-Score:      {f1:.4f} (Macro Avg)')
    print(f'  F-beta (b={beta_val}): {f_beta:.4f} (Macro Avg)')
    print(f'  Inference (ms): mean {inference_ms.mean():.2f} | '
          f'min {inference_ms.min():.2f} | max {inference_ms.max():.2f} | '
          f'p95 {np.percentile(inference_ms, 95):.2f}')
    print(f'  SOS filter (ms): mean {filter_ms.mean():.2f}')
    print(f'  Throughput:    {throughput:.2f} inferences/second')
    print('==================================================')


def save_results(model, y_true, y_predicted, inference_us, filter_us):
    results_directory.mkdir(parents=True, exist_ok=True)
    output_file = results_directory / f'{model}_serial.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['beat_index', 'true_class', 'predicted_class',
                         'inference_us', 'filter_us'])
        for index, row in enumerate(zip(y_true, y_predicted, inference_us, filter_us)):
            writer.writerow([index, *row])
    print(f'\n[DONE] Results saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Benchmark a model running on the ESP32.')
    parser.add_argument('port', help='Serial port, e.g. COM3 or /dev/ttyUSB0')
    parser.add_argument('model', choices=('cnn', 'mlp', 'rf', 'svm'),
                        help='Model currently flashed on the board (used for the output file)')
    parser.add_argument('--beats', type=int, default=200, help='Number of beats to send')
    parser.add_argument('--baud', type=int, default=baud_rate)
    arguments = parser.parse_args()

    Xtest, rrtest, ytest = load_test_set()
    X_selected, rr_selected, y_selected = select_beats(Xtest, rrtest, ytest, arguments.beats)
    print(f'[DATA] Sending {len(X_selected)} beats to {arguments.port}')

    predictions = []
    inference_us = []
    filter_us = []

    with serial.Serial(arguments.port, arguments.baud, timeout=1) as connection:
        # Reset the board so the run always starts from a known state.
        connection.dtr = False
        connection.rts = True
        time.sleep(0.1)
        connection.rts = False
        time.sleep(0.5)

        print('[SERIAL] Waiting for the firmware. Press the RESET button...')
        model_info = wait_for_ready(connection)

        start_time = time.time()
        for index, (beat, rr) in enumerate(zip(X_selected, rr_selected), start=1):
            prediction, inference, filtering = send_beat(connection, beat, rr)
            predictions.append(prediction)
            inference_us.append(inference)
            filter_us.append(filtering)
            if index % 20 == 0 or index == len(X_selected):
                print(f'  {index}/{len(X_selected)} beats sent...')
        total_time = time.time() - start_time

    predictions = np.array(predictions)
    print(f'\n[SERIAL] Finished in {total_time:.1f} s '
          f'({total_time / len(predictions) * 1000:.0f} ms per beat including transfer)')

    print_summary(arguments.model, y_selected, predictions, inference_us, filter_us, model_info)
    save_results(arguments.model, y_selected, predictions, inference_us, filter_us)


if __name__ == '__main__':
    main()