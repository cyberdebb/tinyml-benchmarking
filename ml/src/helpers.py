import os

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import matplotlib.pyplot as plt
import numpy as np
from load_data import RR_FEATURES, aami_mapping, classes, sampling_rate
from keras import models
from sklearn.metrics import accuracy_score
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def print_results(config, model, Xval, yval, classes):
    model.load_weights(config.trained_model)
    
    ypred_mat = model.predict(Xval)

    yval = np.asarray(yval)
    # yval holds one-hot rows for a categorical_crossentropy model, or plain
    # class indices (1D) for a sparse_categorical_crossentropy model.
    ytrue = np.argmax(yval, axis=1) if yval.ndim > 1 else yval.astype(int)
    ypred = np.argmax(ypred_mat, axis=1)
    
    accuracy = accuracy_score(ytrue, ypred)
    print(f"Accuracy: {accuracy:.4f}")


def plot_beat(beat_array, class_id=None):
    """
    Plota um único batimento cardíaco segmentado.
    
    Parâmetros:
    beat_array: array 1D com o sinal do batimento
    class_id: (opcional) ID da classe (0 a 4) para exibir no título
    """
    # Mapeamento reverso para exibir o nome da classe legível
    class_names = {
        0: "Normal (N)",
        1: "Supraventricular (S)",
        2: "Ventricular (V)",
        3: "Fusão (F)",
        4: "Desconhecido (Q)"
    }
    
    plt.figure(figsize=(8, 4))
    
    # Plota o sinal
    plt.plot(beat_array, color='#1f77b4', linewidth=2)
    
    # Configura o título com base na classe fornecida
    title = "Batimento Segmentado"
    if class_id is not None and class_id in class_names:
        title += f" - {class_names[class_id]}"
        
    plt.title(title)
    plt.xlabel("Amostras da Janela")
    plt.ylabel("Amplitude Normalizada")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Desenha uma linha vertical no centro indicando onde está o pico R
    center = len(beat_array) // 2
    plt.axvline(x=center, color='red', linestyle='--', label='Pico R (Centro)', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def numeric_value(values, default=0.0):
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    return float(numeric_values[-1]) if len(numeric_values) else default


def beat_features(processed_signal, beat_time):
    # 0.6 s around the R peak (216 samples at 360 Hz): P wave, QRS and most
    # of the T wave. Must match kFeatureWindow in tinyml_app_{mlp,rf,svm}.cc.
    beat_window_seconds = 0.6
    beat_index = int(round(beat_time * sampling_rate))
    half_window = int(beat_window_seconds * sampling_rate / 2)
    start = max(0, beat_index - half_window)
    end = min(len(processed_signal), beat_index + half_window)
    signal = pd.to_numeric(processed_signal['ECG_Clean'], errors='coerce').dropna().to_numpy()
    window = signal[start:end]

    if len(window) == 0:
        return [0.0] * 12

    centered_window = window - np.mean(window)
    derivative = np.diff(window)
    return [
        float(np.mean(window)),
        float(np.std(window)),
        float(np.min(window)),
        float(np.max(window)),
        float(np.ptp(window)),
        float(np.median(window)),
        float(np.mean(window ** 2)),
        float(np.mean(np.abs(centered_window))),
        float(np.max(np.abs(derivative))) if len(derivative) else 0.0,
        float(np.percentile(window, 10)),
        float(np.percentile(window, 90)),
        float(np.argmax(window) / len(window)),
    ]


def extract_neurokit_features(signals, rr, labels):
    """12 morphology features per beat (beat_features) followed by its RR
    intervals (load_data.RR_FEATURES). Same order as extract_features() +
    the RR values in tinyml_app_{mlp,rf,svm}.cc."""
    feature_rows = []
    total_signals = len(signals)

    print(f'[FEATURES] Starting feature extraction for {total_signals} signals...')

    for signal_index, signal in enumerate(signals, start=1):
        if signal_index == 1 or signal_index % 100 == 0 or signal_index == total_signals:
            print(f'[FEATURES] Processing signal {signal_index}/{total_signals}...')
        signal = np.asarray(signal, dtype=np.float32)

        # `signals` is already the same causal SOS band-pass filtered beat
        # the firmware computes on-device (filter_beats() in load_data.py,
        # filter_sos() on the ESP32). No extra cleaning is applied here on
        # top of that -- an extra step like NeuroKit's ecg_clean() would
        # make beat_features() see something the firmware never produces,
        # a train/inference mismatch the device can't reproduce.
        processed_signal = pd.DataFrame({'ECG_Clean': signal})

        feature_rows.append(beat_features(processed_signal, len(signal) / (2 * sampling_rate)))

    features = np.hstack([
        np.asarray(feature_rows, dtype=np.float32).reshape(len(signals), -1),
        np.asarray(rr, dtype=np.float32).reshape(len(signals), len(RR_FEATURES)),
    ])
    labels = np.asarray(labels)
    print(f'[FEATURES] Extraction completed. Feature matrix shape: {features.shape}')
    print(f'[FEATURES] Labels shape: {labels.shape}')
    return features, labels



def smoothed_class_weights(y):
    """Class weights shared by the four models: sqrt(n_largest / n_class).

    'balanced' weights (n_largest / n_class) give the rarest classes weights
    in the hundreds, and the models then call many normal beats S/F/Q (on
    the inter-patient test, accuracy fell below always predicting N). The
    square root still favors the rare classes, just less aggressively.
    """
    class_ids, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return {int(c): float(np.sqrt(counts.max() / n)) for c, n in zip(class_ids, counts)}


def build_representative_dataset(inputs, n_samples=500, seed=1):
    """Calibration samples for the TFLite int8 quantization.

    inputs is a list with the single input array, or, for a model with
    several inputs, a dict {input name: array}: the converted model orders
    its inputs by name, not in the Keras order, so they are fed by name.
    """
    arrays = list(inputs.values()) if isinstance(inputs, dict) else inputs
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(arrays[0]), size=min(n_samples, len(arrays[0])), replace=False)

    def representative_dataset():
        for index in indices:
            if isinstance(inputs, dict):
                yield {name: array[index:index + 1].astype(np.float32)
                       for name, array in inputs.items()}
            else:
                yield [array[index:index + 1].astype(np.float32) for array in inputs]

    return representative_dataset


def evaluate_tflite(tflite_path, inputs, y):
    """Runs the quantized model on the given inputs (one array per model
    input) to check the accuracy after quantization (this is what actually
    runs on the boards). Each TFLite input is matched to the array with the
    same number of values per sample, like the firmware does."""
    import tensorflow as tf

    print('[EVALUATION] Evaluating quantized TFLite model...')
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()[0]

    feeds = []
    for details in interpreter.get_input_details():
        size = int(np.prod(details['shape'][1:]))
        array = next(a for a in inputs if int(np.prod(a.shape[1:])) == size)
        feeds.append((details, array))

    predictions = np.empty(len(y), dtype=np.int64)
    for index in range(len(y)):
        for details, array in feeds:
            sample = array[index:index + 1].astype(np.float32).reshape(details['shape'])
            if details['dtype'] == np.int8:
                scale, zero_point = details['quantization']
                sample = np.clip(np.round(sample / scale + zero_point), -128, 127).astype(np.int8)
            interpreter.set_tensor(details['index'], sample)
        interpreter.invoke()
        predictions[index] = np.argmax(interpreter.get_tensor(output_details['index'])[0])

    accuracy = accuracy_score(np.asarray(y).astype(int), predictions)
    print(f'[EVALUATION] TFLite int8 accuracy: {accuracy:.4f}')

    ops = sorted({op['op_name'] for op in interpreter._get_ops_details()})
    print(f'[EVALUATION] TFLite ops used (must be registered in the firmware): {ops}')
    return accuracy
