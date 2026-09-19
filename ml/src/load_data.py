import json
import wfdb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
from scipy.signal import butter, sosfilt

# Local storage for the MIT-BIH files and the processed dataset. Based on this
# file's own location (not on the current working directory) so it resolves
# the same way whether a classifier is run directly (e.g. from
# src/classifiers/) or indirectly through pipeline.py.
project_directory = Path(__file__).resolve().parent.parent
data_directory = project_directory / 'data' / 'mitdb'
cache_directory = project_directory / 'data' / 'cache'
record_list_cache = data_directory / 'RECORDS.json'

# 1. Filtro Causal (Pronto para o Edge AI / ESP32)
#
# Aplicado independentemente em cada janela de batimento (256 amostras), a
# partir do estado zero, para reproduzir exatamente o que o firmware faz em
# tinyml_app.cc (filter_sos): o ESP32 recebe o ECG cru e filtra cada janela
# isoladamente, sem contexto do sinal continuo. Por isso o filtro NAO e
# aplicado no sinal continuo do registro nem fica salvo no cache do dataset
# -- ele so entra na hora de treinar/avaliar, para que treino e inferencia no
# device vejam exatamente a mesma transformacao.
def filter_beats(beats, lowcut=0.5, highcut=45.0, fs=360.0, order=4):
    print(f'[FILTER] Applying band-pass filter to {len(beats)} beat window(s)...')
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_beats = sosfilt(sos, beats, axis=-1)
    print('[FILTER] Band-pass filtering completed.')
    return filtered_beats

# 2. Mapeamento AAMI 
classes = ['N', 'S', 'V', 'F', 'Q']
aami_mapping = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    '/': 4, 'f': 4, 'Q': 4,
}

# 2.1 Cache local dos registros do MIT-BIH (baixa uma vez, reusa nas próximas execuções)
def get_mitdb_records():
    """Returns the list of MIT-BIH record names, caching it locally so it only
    has to be fetched from PhysioNet once."""
    if record_list_cache.exists():
        print(f'[CACHE] Using local record list from {record_list_cache}.')
        return json.loads(record_list_cache.read_text())

    print('[DOWNLOAD] Fetching record list from PhysioNet...')
    records = wfdb.get_record_list('mitdb')
    data_directory.mkdir(parents=True, exist_ok=True)
    record_list_cache.write_text(json.dumps(records))
    print(f'[CACHE] Saved record list to {record_list_cache}.')
    return records


def ensure_record_local(record_name):
    """Downloads a MIT-BIH record (signal + annotation) to data_directory the
    first time it is needed. Later calls, from this or any other classifier
    script, reuse the local copy instead of hitting PhysioNet again."""
    header_path = data_directory / f'{record_name}.hea'
    annotation_path = data_directory / f'{record_name}.atr'

    if header_path.exists() and annotation_path.exists():
        return

    print(f'[DOWNLOAD] Record {record_name} not found locally, downloading from PhysioNet...')
    data_directory.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        'mitdb',
        dl_dir=str(data_directory),
        records=[record_name],
        annotators=['atr'],
        keep_subdirs=False,
    )
    print(f'[DOWNLOAD] Record {record_name} saved locally to {data_directory}.')


# 3. Extração por paciente
def load_and_segment_record(record_name, config):
    print(f'[RECORD] Loading record {record_name}...')
    ensure_record_local(record_name)
    local_record_path = str(data_directory / record_name)
    record = wfdb.rdrecord(local_record_path)
    annotation = wfdb.rdann(local_record_path, 'atr')

    # Busca dinamicamente a feature configurada (ex: 'MLII')
    if config.feature in record.sig_name:
        channel_idx = record.sig_name.index(config.feature)
    else:
        print(f'[RECORD] Record {record_name} does not contain feature {config.feature}.')
        return np.array([]), np.array([])
        
    raw_signal = record.p_signal[:, channel_idx]

    # Define o raio da janela com base no input_size (256 // 2 = 128)
    window_radius = config.input_size // 2

    X, y = [], []

    for i in range(len(annotation.sample)):
        peak_idx = annotation.sample[i]
        symbol = annotation.symbol[i]

        if symbol in aami_mapping:
            # Garante que não vai estourar o limite do array
            if peak_idx >= window_radius and peak_idx + window_radius <= len(raw_signal):
                # O tamanho recortado será exatamente config.input_size. Cru:
                # sem filtro e sem normalização -- é exatamente o que o
                # firmware recebe por serial e filtra ele mesmo antes de
                # inferir (ver esp32_firmware/src/tinyml_app.cc).
                beat_window = raw_signal[peak_idx - window_radius : peak_idx + window_radius]

                # Ainda descarta janelas degeneradas (linha reta / sensor
                # travado), que não são um batimento válido de qualquer jeito.
                if beat_window.max() - beat_window.min() > 0:
                    X.append(beat_window)
                    y.append(aami_mapping[symbol])

    X = np.asarray(X)
    y = np.asarray(y)
    print(f'[RECORD] Record {record_name} completed: {len(X)} valid beats extracted (raw, unfiltered).')
    return X, y


# 3.1 Cache local do dataset já processado (janelas extraídas de todos os registros)
def _dataset_cache_path(config):
    return cache_directory / f'dataset_{config.feature}_{config.input_size}.npz'


# 4. Função principal de carregamento
def build_full_dataset(config):
    print('[START] Dataset loading started.')
    print(f"[CONFIG] Feature: {config.feature}")
    print(f"[CONFIG] Input window size: {config.input_size}")

    cache_path = _dataset_cache_path(config)
    if cache_path.exists():
        print(f'[CACHE] Loading cached raw dataset from {cache_path}...')
        cached = np.load(cache_path)
        X_total, y_total = cached['X'], cached['y']
        print(f'[CACHE] Loaded cached dataset: X={X_total.shape}, y={y_total.shape}')
    else:
        records = get_mitdb_records()
        all_X, all_y = [], []
        print(f"[CONFIG] Number of records: {len(records)}")

        for record_index, record_name in enumerate(records, start=1):
            print(f'[DATA] Processing record {record_index}/{len(records)}: {record_name}')

            # Loop infinito que só é quebrado quando o download/processamento dá certo
            while True:
                try:
                    X_patient, y_patient = load_and_segment_record(record_name, config)
                    if len(X_patient) > 0:
                        all_X.append(X_patient)
                        all_y.append(y_patient)
                    else:
                        print(f'[DATA] Record {record_name} skipped because no valid beats were found.')

                    # Se chegou aqui sem dar erro, sai do while e vai para o próximo record
                    break

                except Exception as e:
                    print(f'[ERROR] Failed to process record {record_name}: {e}')
                    print('[RETRY] Retrying in 5 seconds...')
                    time.sleep(5)  # Espera 5 segundos antes de tentar de novo para não sobrecarregar o PhysioNet

        X_total = np.concatenate(all_X, axis=0).astype(np.float32)
        y_total = np.concatenate(all_y, axis=0).astype(np.float32)
        print(f'[DATA] Complete dataset shape: X={X_total.shape}, y={y_total.shape}')

        # Cache guarda o sinal CRU (sem filtro, sem normalização). É esse
        # mesmo cache que o benchmark_serial.py usa para montar o conjunto de
        # validação enviado ao ESP32 -- assim o dispositivo recebe o mesmo
        # tipo de sinal que receberia em um deployment real (ECG cru de um
        # ADC), e é ele mesmo quem filtra antes de inferir.
        cache_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X_total, y=y_total)
        print(f'[CACHE] Saved raw dataset to {cache_path}.')

    # 3. Usa a config para decidir se faz o split ou não
    if config.split:
        print('[SPLIT] Splitting dataset into training and validation sets...')
        X, Xval, y, yval = train_test_split(X_total, y_total, test_size=0.2, random_state=42)
        print(f'[SPLIT] Training shapes: X={X.shape}, y={y.shape}')
        print(f'[SPLIT] Validation shapes: X={Xval.shape}, y={yval.shape}')

        # O filtro só é aplicado aqui, sobre os splits em memória -- nunca é
        # salvo de volta no cache. Reproduz exatamente o que o firmware faz
        # com cada janela recebida por serial (ver filter_beats acima).
        X = filter_beats(X)
        Xval = filter_beats(Xval)

        print('[DONE] Dataset loading completed.')
        return (X, y, Xval, yval)
    else:
        print('[SPLIT] Dataset split disabled.')
        X_total = filter_beats(X_total)
        print(f'[DATA] Returning complete dataset with shape: {X_total.shape}')
        print('[DONE] Dataset loading completed.')
        return X_total, y_total