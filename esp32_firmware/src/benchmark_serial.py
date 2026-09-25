"""Sends ECG beats to the ESP32 over UART and collects the predictions.

Usage:
    python src/benchmark_serial.py COM3 <model>              # whole DS2 test set
    python src/benchmark_serial.py COM3 <model> --resume     # continue a stopped run
    python src/benchmark_serial.py COM3 <model> --beats 500  # quick check on a sample

By default every beat of the test set (DS2, the same patients the models were
evaluated on in training) is sent, so the metrics printed here are the
model's real test metrics, measured on the board. That takes hours (each
beat is ~2 KB at 115200 baud plus the inference), so every result is
written to the CSV as soon as it arrives: if the run stops (serial error,
Ctrl+C, board reset), run the same command again with --resume and it
continues from the first beat that is not in the CSV yet.

The firmware answers every beat with a line:
    RESULT,<predicted_class>,<inference_microseconds>,<filter_microseconds>

Results are written to results/<model>_serial.csv (per beat) and
results/<model>_report.txt (the summary printed at the end); a --beats run
writes <model>_serial_sample<N>.csv / _report.txt instead, so it never
overwrites a full run.
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
from load_data import TEST_RECORDS, cache_is_current, dataset_cache_path, select_records  # noqa: E402
from metrics import SCORED_CLASS_IDS, aami_report  # noqa: E402

# Must match the classifier configs and the pipeline dataset_config.
dataset_config = SimpleNamespace(feature='MLII', input_size=256)

baud_rate = 115200
# One beat takes a few hundred ms to travel at 115200 baud, so the timeout has
# to cover transfer plus inference.
read_timeout_seconds = 15
class_names = ('N', 'S', 'V', 'F', 'Q')
csv_header = ['beat_index', 'record', 'true_class', 'predicted_class', 'inference_us', 'filter_us']


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
        if not cache_is_current(data):
            print('[ERROR] Dataset cache is from an older version (other RR features or records).')
            print('[ERROR] Run the training (pipeline step 1) again to rebuild it.')
            sys.exit(1)
        X_total = data['X']
        rr_total = data['rr']
        y_total = data['y']
        records = data['records']

    mask = select_records(records, TEST_RECORDS)
    print(f'[DATA] Test set (DS2): {mask.sum()} beats')
    return X_total[mask], rr_total[mask], y_total[mask].astype(int), records[mask]


def select_beats(y, n_beats, seed=1):
    """Indices (into the test set) of the beats to send: all of them, or a
    random subset of n_beats keeping the class distribution. The seed is
    fixed, so a --resume of a --beats run picks the same beats again."""
    if n_beats is None or n_beats >= len(y):
        return np.arange(len(y))
    indices = np.arange(len(y))
    try:
        indices, _ = train_test_split(
            indices, train_size=n_beats, stratify=y, random_state=seed)
    except ValueError:
        # A class too rare to stratify (fewer than 2 beats): plain random subset.
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(y), size=n_beats, replace=False)
    return np.sort(indices)


def sample_format(X):
    """printf format for the beat samples. MIT-BIH samples are ADC counts /
    200 (multiples of 0.005 mV), so 3 decimals send exactly the same values
    as 5 in ~20% fewer bytes -- which is ~20% less time per beat, since the
    transfer takes longer than the inference. Falls back to 5 decimals if
    the signal ever has finer steps."""
    return '.3f' if np.allclose(np.round(X, 3), X, rtol=0, atol=1e-6) else '.5f'


def wait_for_ready(connection, timeout=20):
    """Reads the boot log until the firmware reports it is ready."""
    deadline = time.time() + timeout
    model_info = {}
    while time.time() < deadline:
        line = connection.readline().decode('utf-8', errors='replace').strip()
        if not line:
            continue
        print(f'  [esp32] {line}')
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


def send_beat(connection, beat, rr, beat_format):
    """Sends one beat followed by its RR features (load_data.RR_FEATURES)
    and returns (predicted_class, inference_microseconds, filter_microseconds)."""
    payload = ','.join([format(value, beat_format) for value in beat]
                       + [f'{value:.7g}' for value in rr]) + '\n'
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
        # Anything else is an ESP_LOG line; show it, it usually explains a problem.
        print(f'  [esp32] {line}')
    raise TimeoutError('No RESULT line received from the firmware.')


def read_results(output_file):
    """Rows already in a results CSV (for --resume), as a list of dicts."""
    with open(output_file, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != csv_header:
            print(f'[ERROR] {output_file} has other columns ({reader.fieldnames}); '
                  'it is from an older version of this script. Run without --resume.')
            sys.exit(1)
        return [{key: int(value) for key, value in row.items()} for row in reader]


def summary_text(model, y_true, y_predicted, records, inference_us, filter_us, model_info):
    inference_ms = np.array(inference_us) / 1000.0
    filter_ms = np.array(filter_us) / 1000.0

    # 1. Acurácia
    accuracy = accuracy_score(y_true, y_predicted)

    # Médias macro sobre as mesmas classes do treino (SCORED_CLASSES: N, S,
    # V, F). Q fica de fora: quase nao ha Q no DS2 sem os registros com
    # marcapasso, e uma unica previsao Q errada colocaria na media uma classe
    # com metricas 0.
    avg_method = 'macro'
    labels = SCORED_CLASS_IDS

    # 2. Sensibilidade (Recall)
    sensitivity = recall_score(y_true, y_predicted, labels=labels, average=avg_method, zero_division=0)

    # 3. F1-Score
    f1 = f1_score(y_true, y_predicted, labels=labels, average=avg_method, zero_division=0)

    # 4. F-beta Score (usando beta=2.0 como exemplo para dar mais peso à sensibilidade)
    beta_val = 2.0
    f_beta = fbeta_score(y_true, y_predicted, beta=beta_val, labels=labels,
                         average=avg_method, zero_division=0)

    # 5. Especificidade
    # A especificidade em multiclasse precisa ser calculada manualmente extraindo os Verdadeiros Negativos (TN)
    # e Falsos Positivos (FP) da Matriz de Confusão para cada classe.
    # Fixed labels so the matrix is always 5x5, even when a rare class is
    # missing from the beats sent.
    cm = confusion_matrix(y_true, y_predicted, labels=np.arange(len(class_names)))
    specificities = []
    for i in labels:
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        # Evita divisão por zero
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    # Média das especificidades das classes avaliadas
    specificity = np.mean(specificities)

    # 6. Vazão Computacional (Throughput)
    # Convertendo o tempo médio de inferência para segundos e dividindo 1 por ele
    mean_inference_sec = np.mean(inference_us) / 1_000_000.0
    throughput = 1.0 / mean_inference_sec if mean_inference_sec > 0 else 0.0

    scored = ', '.join(class_names[i] for i in labels)
    lines = ['==================================================',
             f'  Model: {model}']
    if model_info:
        lines += [f"  Model size: {model_info['model_bytes'] / 1024:.1f} KB",
                  f"  Arena used: {model_info['arena_bytes'] / 1024:.1f} KB"]
    lines += [
        f'  Beats: {len(y_true)}',
        f'  Accuracy:      {accuracy:.4f}',
        f'  Sensitivity:   {sensitivity:.4f} (Macro Avg {scored})',
        f'  Specificity:   {specificity:.4f} (Macro Avg {scored})',
        f'  F1-Score:      {f1:.4f} (Macro Avg {scored})',
        f'  F-beta (b={beta_val}): {f_beta:.4f} (Macro Avg {scored})',
        f'  Inference (ms): mean {inference_ms.mean():.2f} | '
        f'min {inference_ms.min():.2f} | max {inference_ms.max():.2f} | '
        f'p95 {np.percentile(inference_ms, 95):.2f}',
        f'  SOS filter (ms): mean {filter_ms.mean():.2f}',
        f'  Throughput:    {throughput:.2f} inferences/second',
        '==================================================',
    ]
    report, _ = aami_report(f'{model} on the board', y_true, y_predicted, records)
    return report + '\n\n' + '\n'.join(lines)


def format_duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f'{hours}h{rest // 60:02d}m'


def main():
    # With the output piped (e.g. '| Tee-Object'), Windows uses its ANSI code
    # page (cp1252) for stdout, and a garbled byte from the board ('\ufffd')
    # would crash print(): replace what can't be encoded instead. Piped
    # stdout is also block-buffered: line_buffering keeps the progress live.
    # (No need to pipe it: the summary is saved to results/<model>_report.txt.)
    sys.stdout.reconfigure(errors='replace', line_buffering=True)

    parser = argparse.ArgumentParser(description='Benchmark a model running on the ESP32.')
    parser.add_argument('port', help='Serial port, e.g. COM3 or /dev/ttyUSB0')
    parser.add_argument('model', choices=('cnn', 'mlp', 'rf', 'svm'),
                        help='Model currently flashed on the board (used for the output file)')
    parser.add_argument('--beats', type=int, default=None,
                        help='Send only a random sample of this many beats (quick check). '
                             'Default: the whole DS2 test set.')
    parser.add_argument('--resume', action='store_true',
                        help='Continue a stopped run: skip the beats already in the results CSV')
    parser.add_argument('--baud', type=int, default=baud_rate)
    arguments = parser.parse_args()

    Xtest, rrtest, ytest, records = load_test_set()
    selected = select_beats(ytest, arguments.beats)
    beat_format = sample_format(Xtest[selected])

    suffix = '' if len(selected) == len(ytest) else f'_sample{len(selected)}'
    results_directory.mkdir(parents=True, exist_ok=True)
    output_file = results_directory / f'{arguments.model}_serial{suffix}.csv'
    report_file = results_directory / f'{arguments.model}_report{suffix}.txt'

    done_rows = []
    if arguments.resume and output_file.exists():
        done_rows = read_results(output_file)
        print(f'[RESUME] {len(done_rows)} beats already in {output_file}')
    elif output_file.exists():
        print(f'[WARNING] Overwriting {output_file} (use --resume to continue it instead)')
    done = {row['beat_index'] for row in done_rows}
    pending = [index for index in selected if index not in done]
    print(f'[DATA] Sending {len(pending)} of {len(selected)} beats to {arguments.port}')

    model_info = {}
    stopped = None
    if pending:
        mode = 'a' if done_rows else 'w'
        with open(output_file, mode, newline='', encoding='utf-8') as handle, \
                serial.Serial(arguments.port, arguments.baud, timeout=1) as connection:
            writer = csv.writer(handle)
            if mode == 'w':
                writer.writerow(csv_header)

            # Reset the board so the run always starts from a known state.
            connection.dtr = False
            connection.rts = True
            time.sleep(0.1)
            connection.rts = False
            time.sleep(0.5)

            print('[SERIAL] Waiting for the firmware...')
            model_info = wait_for_ready(connection)

            start_time = time.time()
            try:
                for count, index in enumerate(pending, start=1):
                    prediction, inference, filtering = send_beat(
                        connection, Xtest[index], rrtest[index], beat_format)
                    row = {'beat_index': int(index), 'record': int(records[index]),
                           'true_class': int(ytest[index]), 'predicted_class': prediction,
                           'inference_us': inference, 'filter_us': filtering}
                    writer.writerow([row[key] for key in csv_header])
                    handle.flush()
                    done_rows.append(row)
                    if count % 100 == 0 or count == len(pending):
                        elapsed = time.time() - start_time
                        remaining = elapsed / count * (len(pending) - count)
                        print(f'  {len(done_rows)}/{len(selected)} beats '
                              f'({elapsed / count * 1000:.0f} ms/beat, '
                              f'~{format_duration(remaining)} left)')
            except (TimeoutError, serial.SerialException, KeyboardInterrupt) as error:
                stopped = error
            total_time = time.time() - start_time

        sent = len(done_rows) - len(done)
        if sent:
            print(f'\n[SERIAL] Sent {sent} beats in {format_duration(total_time)} '
                  f'({total_time / sent * 1000:.0f} ms per beat including transfer)')

    if stopped is not None:
        print(f'\n[STOPPED] {type(stopped).__name__}: {stopped}')
        print(f'[STOPPED] {len(done_rows)}/{len(selected)} beats saved to {output_file}.')
        print('[STOPPED] Run the same command with --resume to continue from there.')
        sys.exit(1)

    rows = sorted(done_rows, key=lambda row: row['beat_index'])
    text = summary_text(
        arguments.model,
        np.array([row['true_class'] for row in rows]),
        np.array([row['predicted_class'] for row in rows]),
        np.array([row['record'] for row in rows]),
        [row['inference_us'] for row in rows],
        [row['filter_us'] for row in rows],
        model_info,
    )
    print('\n' + text)
    report_file.write_text(text + '\n', encoding='utf-8')
    print(f'\n[DONE] Results saved to {output_file} and {report_file}')


if __name__ == '__main__':
    main()
