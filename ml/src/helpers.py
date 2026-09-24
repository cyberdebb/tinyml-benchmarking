import os

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import matplotlib.pyplot as plt
import numpy as np
from load_data import aami_mapping, classes, sampling_rate
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


def extract_neurokit_features(signals, labels):
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

    features = np.asarray(feature_rows, dtype=np.float32)
    labels = np.asarray(labels)
    print(f'[FEATURES] Extraction completed. Feature matrix shape: {features.shape}')
    print(f'[FEATURES] Labels shape: {labels.shape}')
    return features, labels



def build_representative_dataset(X, n_samples=500, seed=1):
    """Calibration samples for the TFLite int8 quantization."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)

    def representative_dataset():
        for index in indices:
            yield [X[index:index + 1].astype(np.float32)]

    return representative_dataset


def evaluate_tflite(tflite_path, X, y):
    """Runs the quantized model on X to check the accuracy after
    quantization (this is what actually runs on the boards)."""
    import tensorflow as tf

    print('[EVALUATION] Evaluating quantized TFLite model...')
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    predictions = np.empty(len(X), dtype=np.int64)

    for index in range(len(X)):
        sample = X[index:index + 1].astype(np.float32)
        if input_details['dtype'] == np.int8:
            sample = np.clip(np.round(sample / input_scale + input_zero_point), -128, 127)
            sample = sample.astype(np.int8)
        interpreter.set_tensor(input_details['index'], sample)
        interpreter.invoke()
        predictions[index] = np.argmax(interpreter.get_tensor(output_details['index'])[0])

    accuracy = accuracy_score(np.asarray(y).astype(int), predictions)
    print(f'[EVALUATION] TFLite int8 accuracy: {accuracy:.4f}')

    ops = sorted({op['op_name'] for op in interpreter._get_ops_details()})
    print(f'[EVALUATION] TFLite ops used (must be registered in the firmware): {ops}')
    return accuracy
