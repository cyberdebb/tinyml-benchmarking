import wfdb
import numpy as np
from pathlib import Path
import time
from scipy.signal import butter, sosfilt

# Local storage for the MIT-BIH files and the processed dataset. Based on this
# file's own location (not on the current working directory) so it resolves
# the same way whether a classifier is run directly (e.g. from
# src/classifiers/) or indirectly through pipeline.py.
project_directory = Path(__file__).resolve().parent.parent
data_directory = project_directory / 'data' / 'mitdb'
cache_directory = project_directory / 'data' / 'cache'

# Frequencia de amostragem do MIT-BIH (Hz)
sampling_rate = 360

# 0. Divisao inter-paciente (de Chazal et al., 2004)
#
# Um split aleatorio por batimento coloca batimentos do mesmo paciente no
# treino e no teste, e a acuracia sai inflada (o modelo reconhece o paciente,
# nao a arritmia). Aqui cada registro (paciente) fica inteiro em um unico
# conjunto:
#   - DS1: treino. Uma parte dos registros e separada como validacao (early
#     stopping / escolha do modelo).
#   - DS2: teste. Nunca e usado no treino nem na escolha do modelo; e o
#     conjunto das metricas finais e do benchmark nas placas
#     (benchmark_serial.py).
# Os registros com marcapasso (102, 104, 107, 217) ficam de fora, como
# recomenda a AAMI.
DS1_RECORDS = (
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
    '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230',
)
TEST_RECORDS = (
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
    '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234',
)
# Registros do DS1 usados como validacao (~20% dos batimentos do DS1). Foram
# escolhidos para que a validacao tenha batimentos S, V e F sem tirar do
# treino os registros que concentram essas classes (208 tem quase todos os F
# do DS1, 209 boa parte dos S). A distribuicao de classes de cada conjunto e
# impressa em build_full_dataset() para conferir.
VALIDATION_RECORDS = ('118', '124', '205', '223')
TRAIN_RECORDS = tuple(r for r in DS1_RECORDS if r not in VALIDATION_RECORDS)

# 1. Filtro Causal (Pronto para o Edge AI / ESP32)
#
# Aplicado independentemente em cada janela de batimento (256 amostras), para
# reproduzir exatamente o que o firmware faz em tinyml_app_<modelo>.cc
# (filter_sos): a placa recebe o ECG cru e filtra cada janela isoladamente,
# sem contexto do sinal continuo. Por isso o filtro NAO e aplicado no sinal
# continuo do registro nem fica salvo no cache do dataset -- ele so entra na
# hora de treinar/avaliar, para que treino e inferencia no device vejam
# exatamente a mesma transformacao.
#
# Antes de filtrar, o nivel da primeira amostra e subtraido da janela. Sem
# isso, o filtro parte do estado zero e o offset DC da janela (linha de base)
# vira um transiente que atravessa a janela inteira. Como o ganho DC do
# passa-banda e zero, subtrair x[0] e filtrar a partir do zero e exatamente
# igual a filtrar o sinal original partindo do regime permanente para x[0]
# (sosfilt_zi(sos) * x[0]) -- e custa so uma subtracao no firmware.
def filter_beats(beats, lowcut=0.5, highcut=45.0, fs=sampling_rate, order=4):
    print(f'[FILTER] Applying band-pass filter to {len(beats)} beat window(s)...')
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    beats = np.asarray(beats)
    filtered_beats = sosfilt(sos, beats - beats[..., :1], axis=-1)
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
                # inferir (ver tinyml_app_<modelo>.cc).
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
def dataset_cache_path(feature, input_size):
    return cache_directory / f'dataset_{feature}_{input_size}.npz'


def load_cached_dataset(config):
    """Returns (X, y, records) from the raw dataset cache, building it first
    if needed. records holds the MIT-BIH record number of every beat, which
    is what the inter-patient split is based on."""
    cache_path = dataset_cache_path(config.feature, config.input_size)
    if cache_path.exists():
        print(f'[CACHE] Loading cached raw dataset from {cache_path}...')
        with np.load(cache_path) as cached:
            if 'records' in cached.files:
                X_total, y_total, records = cached['X'], cached['y'], cached['records']
                print(f'[CACHE] Loaded cached dataset: X={X_total.shape}, y={y_total.shape}')
                return X_total, y_total, records
        # Caches from before the inter-patient split don't know which record
        # each beat came from, so they can't be split by patient.
        print('[CACHE] Cached dataset has no record ids (old format), rebuilding it...')

    records_to_load = DS1_RECORDS + TEST_RECORDS
    all_X, all_y, all_records = [], [], []
    print(f"[CONFIG] Number of records: {len(records_to_load)}")

    for record_index, record_name in enumerate(records_to_load, start=1):
        print(f'[DATA] Processing record {record_index}/{len(records_to_load)}: {record_name}')

        # Loop infinito que só é quebrado quando o download/processamento dá certo
        while True:
            try:
                X_patient, y_patient = load_and_segment_record(record_name, config)
                if len(X_patient) > 0:
                    all_X.append(X_patient)
                    all_y.append(y_patient)
                    all_records.append(np.full(len(X_patient), int(record_name), dtype=np.int16))
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
    records = np.concatenate(all_records, axis=0)
    print(f'[DATA] Complete dataset shape: X={X_total.shape}, y={y_total.shape}')

    # Cache guarda o sinal CRU (sem filtro, sem normalização). É esse
    # mesmo cache que o benchmark_serial.py usa para montar o conjunto de
    # teste enviado às placas -- assim o dispositivo recebe o mesmo tipo de
    # sinal que receberia em um deployment real (ECG cru de um ADC), e é ele
    # mesmo quem filtra antes de inferir.
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X_total, y=y_total, records=records)
    print(f'[CACHE] Saved raw dataset to {cache_path}.')
    return X_total, y_total, records


def select_records(X, y, records, record_names):
    """Keeps only the beats that belong to the given records."""
    mask = np.isin(records, [int(name) for name in record_names])
    return X[mask], y[mask]


def print_class_distribution(name, y):
    counts = np.bincount(np.asarray(y).astype(int), minlength=len(classes))
    summary = ', '.join(f'{label}={count}' for label, count in zip(classes, counts))
    print(f'[SPLIT] {name}: {len(y)} beats ({summary})')


# 4. Função principal de carregamento
def build_full_dataset(config):
    """With config.split, returns (X_train, y_train, X_val, y_val, X_test,
    y_test), split by patient (see DS1_RECORDS / TEST_RECORDS above);
    otherwise returns every beat as (X, y). Beats come band-pass filtered."""
    print('[START] Dataset loading started.')
    print(f"[CONFIG] Feature: {config.feature}")
    print(f"[CONFIG] Input window size: {config.input_size}")

    X_total, y_total, records = load_cached_dataset(config)

    # 3. Usa a config para decidir se faz o split ou não
    if config.split:
        print('[SPLIT] Splitting dataset by patient (inter-patient, de Chazal DS1/DS2)...')
        X, y = select_records(X_total, y_total, records, TRAIN_RECORDS)
        Xval, yval = select_records(X_total, y_total, records, VALIDATION_RECORDS)
        Xtest, ytest = select_records(X_total, y_total, records, TEST_RECORDS)
        print_class_distribution('Training (DS1)', y)
        print_class_distribution('Validation (DS1)', yval)
        print_class_distribution('Test (DS2)', ytest)

        # O filtro só é aplicado aqui, sobre os splits em memória -- nunca é
        # salvo de volta no cache. Reproduz exatamente o que o firmware faz
        # com cada janela recebida por serial (ver filter_beats acima).
        X = filter_beats(X)
        Xval = filter_beats(Xval)
        Xtest = filter_beats(Xtest)

        print('[DONE] Dataset loading completed.')
        return X, y, Xval, yval, Xtest, ytest
    else:
        print('[SPLIT] Dataset split disabled.')
        X_total = filter_beats(X_total)
        print(f'[DATA] Returning complete dataset with shape: {X_total.shape}')
        print('[DONE] Dataset loading completed.')
        return X_total, y_total
