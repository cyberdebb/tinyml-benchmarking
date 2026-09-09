import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import butter, sosfilt

# 1. Filtro Causal (Pronto para o Edge AI / ESP32)
def realtime_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data)

# 2. Mapeamento AAMI 
classes = ['N', 'A', 'V', 'F', 'P']
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'P': 4, '/': 4, 'f': 4, 'u': 4
}

# 3. Extração por paciente
def load_and_segment_record(record_name, config):
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')
    annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    
    # Busca dinamicamente a feature configurada (ex: 'MLII')
    if config.feature in record.sig_name:
        channel_idx = record.sig_name.index(config.feature)
    else:
        # Se o paciente não tiver a derivação MLII, ignoramos ele
        return np.array([]), np.array([])
        
    raw_signal = record.p_signal[:, channel_idx]
    clean_signal = realtime_bandpass_filter(raw_signal)
    
    # Define o raio da janela com base no input_size (256 // 2 = 128)
    window_radius = config.input_size // 2
    
    X, y = [], []
    
    for i in range(len(annotation.sample)):
        peak_idx = annotation.sample[i]
        symbol = annotation.symbol[i]
        
        if symbol in aami_mapping:
            # Garante que não vai estourar o limite do array
            if peak_idx >= window_radius and peak_idx + window_radius <= len(clean_signal):
                # O tamanho recortado será exatamente config.input_size
                beat_window = clean_signal[peak_idx - window_radius : peak_idx + window_radius]
                
                beat_min = np.min(beat_window)
                beat_max = np.max(beat_window)
                
                if beat_max - beat_min > 0:
                    beat_normalized = (beat_window - beat_min) / (beat_max - beat_min)
                    X.append(beat_normalized)
                    y.append(aami_mapping[symbol])
            
    return np.array(X), np.array(y)


# 4. Função principal de carregamento
def build_full_dataset(config):
    records = wfdb.get_record_list('mitdb')
    all_X, all_y = [], []
    
    print(f"Extraindo sinais da derivação '{config.feature}' com janela de {config.input_size} amostras...")
    
    for record_name in records:
        try:
            X_patient, y_patient = load_and_segment_record(record_name, config)
            if len(X_patient) > 0:
                all_X.append(X_patient)
                all_y.append(y_patient)
            else:
                print(f"[Paciente {record_name} ignorado: Não possui a derivação {config.feature}]")
        except Exception as e:
            print(f"[Erro no paciente {record_name}: {e}]")
            
    X_total = np.concatenate(all_X, axis=0).astype(np.float32)
    y_total = np.concatenate(all_y, axis=0).astype(np.float32)
    
    # 3. Usa a config para decidir se faz o split ou não
    if config.split:
        print("Embaralhando e separando em treino e validação...")
        X, Xval, y, yval = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
        print(f"Treino: {X.shape} | Validação: {Xval.shape}")
        return (X, y, Xval, yval)
    else:
        print("Split desativado. Retornando matriz unificada...")
        print(f"Dataset Total: {X_total.shape}")
        return X_total, y_total