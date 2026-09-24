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
cache_directory = dataset_directory / 'data' / 'cache'
firmware_directory = Path(__file__).resolve().parents[1]
results_directory = firmware_directory / 'results'

# Must match the classifier configs and the pipeline dataset_config.
dataset_config = SimpleNamespace(feature='MLII', input_size=256)
split_random_state = 42
split_test_size = 0.2

baud_rate = 115200
# One beat takes a few hundred ms to travel at 115200 baud, so the timeout has
# to cover transfer plus inference.
read_timeout_seconds = 15
class_names = ('N', 'S', 'V', 'F', 'Q')


def load_validation_set():
    """Loads the same validation split every classifier was evaluated on.

    The cache holds the raw, unfiltered ECG windows (see ml/src/load_data.py):
    the band-pass filter is applied only at training/evaluation time on the
    host, never saved back to the cache. So the beats sent here over serial
    are raw, matching what the firmware itself filters on-device
    (tinyml_app_<model>.cc, filter_sos) before running inference.
    """
    cache_file = cache_directory / f'dataset_{dataset_config.feature}_{dataset_config.input_size}.npz'
    if not cache_file.exists():
        print(f'[ERROR] Dataset cache not found: {cache_file}')
        print('[ERROR] Run the pipeline (step 0) first.')
        sys.exit(1)

    with np.load(cache_file) as data:
        X_total = data['X']
        y_total = data['y']

    _, Xval, _, yval = train_test_split(
        X_total,
        y_total,
        test_size=split_test_size,
        random_state=split_random_state,
    )
    print(f'[DATA] Validation set: {Xval.shape[0]} beats')
    return Xval, yval.astype(int)


def select_beats(Xval, yval, n_beats, seed=1):
    """Picks a random subset, keeping the class distribution of the full set."""
    if n_beats >= len(Xval):
        return Xval, yval
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(Xval), size=n_beats, replace=False)
    return Xval[indices], yval[indices]


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


def send_beat(connection, beat):
    """Sends one beat and returns (predicted_class, inference_microseconds)."""
    payload = ','.join(f'{value:.5f}' for value in beat) + '\n'
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
    cm = confusion_matrix(y_true, y_predicted)
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

    Xval, yval = load_validation_set()
    X_selected, y_selected = select_beats(Xval, yval, arguments.beats)
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

        print('[SERIAL] Waiting for the firmware...')
        model_info = wait_for_ready(connection)

        start_time = time.time()
        for index, beat in enumerate(X_selected, start=1):
            prediction, inference, filtering = send_beat(connection, beat)
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