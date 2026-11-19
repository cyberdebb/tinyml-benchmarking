import wfdb
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import time
from scipy.signal import butter, sosfilt

# Local storage for the MIT-BIH files and the processed dataset. Based on this
# file's own location (not on the current working directory) so it resolves
# the same way whether a classifier is run directly (e.g. from
# src/classifiers/) or indirectly through pipeline.py.
project_directory = Path(__file__).resolve().parent.parent
data_directory = project_directory / 'data' / 'mitdb'
cache_directory = project_directory / 'data' / 'cache'

# MIT-BIH sampling frequency (Hz)
sampling_rate = 360

# 0. Inter-patient split (Chazal et al., 2004)
#
# A random split by beat places beats from the same patient in both training
# and test sets, inflating accuracy (the model recognizes the patient, not the
# arrhythmia). Here each record (patient) remains entirely in one split:
#   - DS1: training. Some records are held out for validation (early stopping
#     / model selection).
#   - DS2: test. Never used for training or model selection; it is the set for
#     final metrics and board benchmarks (benchmark_serial.py).
#
# Paced records (102, 104, 107, 217): AAMI (EC57) and Chazal exclude them,
# and INCLUDE_PACED_RECORDS = False follows that protocol so results remain
# comparable with the literature. Without them, class Q has only a few 'Q'
# beats from other records: there are too few to learn or evaluate Q, and the
# main metrics are for N, S, V, and F (SCORED_CLASSES). With True, the two
# paced records with MLII are included, one on each side (217 in training, 107
# in testing; 102 and 104 have no MLII). However, class Q then comes entirely
# from one patient in each split, so Q measures only how much 107 resembles 217.
INCLUDE_PACED_RECORDS = False

DS1_RECORDS = (
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
    '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230',
)
DS2_RECORDS = (
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
    '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234',
)
# DS1 records used for validation (~20% of DS1 beats). They were selected so
# validation contains S, V, and F beats without removing from training the
# records that concentrate these classes (208 has nearly all DS1 F beats, 209
# many of the S beats). Each split's class distribution is printed by
# build_full_dataset() for verification.
VALIDATION_RECORDS = ('118', '124', '205', '223')
PACED_TRAIN_RECORDS = ('217',)
PACED_TEST_RECORDS = ('107',)
TRAIN_RECORDS = tuple(r for r in DS1_RECORDS if r not in VALIDATION_RECORDS) + (
    PACED_TRAIN_RECORDS if INCLUDE_PACED_RECORDS else ())
TEST_RECORDS = DS2_RECORDS + (PACED_TEST_RECORDS if INCLUDE_PACED_RECORDS else ())
# The cache always stores all records (including paced records); the splits
# above only select which ones to use, so changing INCLUDE_PACED_RECORDS does
# not require rebuilding the cache.
ALL_RECORDS = DS1_RECORDS + PACED_TRAIN_RECORDS + DS2_RECORDS + PACED_TEST_RECORDS

# RR features for each beat, based on the annotated R peaks (the same peaks
# that center the beat window). The waveform alone does not show that a beat
# arrived early, which is what most separates S (and helps V/F) from N. The
# host sends these values with each beat (benchmark_serial.py); the firmware
# does not calculate RR.
#
# All are ratios, not intervals in seconds: absolute RR depends on each
# patient's heart rate, and an inter-patient split would otherwise make the
# model learn the training patients' rhythm instead of prematurity. What goes
# to the models is the LOG of each ratio: 0 = normal rhythm, symmetric between
# early and late (0.5 and 2 become -0.69 and +0.69), without the long tails of
# raw ratios (a 6 s pause gives post_pre_rr ~8). Long tails harm int8
# quantization: the MLP input and the CNN RR input each have one scale for the
# entire tensor, and an extreme value leaves few levels for typical values.
# Averages use only intervals BEFORE the beat (causal, as in real time, and
# without a premature beat pulling its own reference):
#   local_rr: mean of the last RR_LOCAL_BEATS intervals (recent rhythm)
#   long_rr:  mean of the last RR_LONG_BEATS intervals (~5 min, patient's
#             baseline rhythm)
# and the features are:
#   log_pre_rr_local:  log(pre_rr / local_rr)  (prematurity)
#   log_post_rr_local: log(post_rr / local_rr) (pause after the beat)
#   log_post_pre_rr:   log(post_rr / pre_rr)   (compensatory pause)
#   log_pre_rr_long:   log(pre_rr / long_rr)   (prematurity relative to
#                      baseline rhythm, robust when the last beats
#                      were also ectopic, as in bigeminy)
# where pre_rr = current R - previous R and post_rr = next R - current R.
RR_FEATURES = ('log_pre_rr_local', 'log_post_rr_local', 'log_post_pre_rr', 'log_pre_rr_long')
RR_LOCAL_BEATS = 10
RR_LONG_BEATS = 300

# Classes used to compare models (macro-F1, checkpoint selection).
# Q is excluded: without paced records there is almost no Q, and with them Q
# comes from one patient in each split (see INCLUDE_PACED_RECORDS).
SCORED_CLASSES = ('N', 'S', 'V', 'F')

# Cache format version: increment whenever its contents change (windows,
# RR_FEATURES, records), so old caches are rebuilt.
DATASET_CACHE_VERSION = 3

# 1. Causal filter (ready for Edge AI / ESP32)
#
# Applied independently to each beat window (256 samples) to exactly reproduce
# what the firmware does in model_<model>.cc (filter_sos): the board receives
# raw ECG and filters each window independently, without continuous-signal
# context. Therefore the filter is not applied to the continuous record or
# saved in the dataset cache; it is applied only during training/evaluation so
# training and device inference see exactly the same transformation.
#
# Before filtering, the first sample level is subtracted from the window.
# Otherwise the filter starts at zero state and the window's DC offset (baseline)
# becomes a transient spanning the entire window. Since the band-pass DC gain
# is zero, subtracting x[0] and filtering from zero is exactly equivalent to
# filtering the original signal from steady state at x[0] (sosfilt_zi(sos) *
# x[0]), and costs only one subtraction in the firmware.
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


# 3. Per-patient extraction
def load_and_segment_record(record_name, config):
    print(f'[RECORD] Loading record {record_name}...')
    ensure_record_local(record_name)
    local_record_path = str(data_directory / record_name)
    record = wfdb.rdrecord(local_record_path)
    annotation = wfdb.rdann(local_record_path, 'atr')

    # Dynamically find the configured feature (e.g. 'MLII')
    if config.feature in record.sig_name:
        channel_idx = record.sig_name.index(config.feature)
    else:
        print(f'[RECORD] Record {record_name} does not contain feature {config.feature}.')
        return np.array([]), np.array([]), np.array([])
        
    raw_signal = record.p_signal[:, channel_idx]

    # Define the window radius from input_size (256 // 2 = 128)
    window_radius = config.input_size // 2

    # R peaks of beats (non-beat annotations, such as '+' and '~', are
    # excluded); these are the basis for both windows and RR intervals.
    beat_positions = [
        (sample, symbol)
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in aami_mapping
    ]
    peaks = np.array([sample for sample, _ in beat_positions], dtype=np.float64)
    pre_rr_all = np.diff(peaks, prepend=np.nan) / record.fs

    X, rr, y = [], [], []

    # The first and last beats have no previous/next RR interval.
    for i in range(1, len(beat_positions) - 1):
        peak_idx, symbol = beat_positions[i]

        # Ensure the window stays within the array bounds.
        if peak_idx >= window_radius and peak_idx + window_radius <= len(raw_signal):
            # The extracted size is exactly config.input_size. Raw, without
            # filtering or normalization, this is exactly what the firmware
            # receives over serial and filters before inference
            # (see model_<model>.cc).
            beat_window = raw_signal[peak_idx - window_radius : peak_idx + window_radius]

            # Also discard degenerate windows (flat line / stalled sensor),
            # which are not valid beats anyway.
            if beat_window.max() - beat_window.min() > 0:
                X.append(beat_window)
                rr.append(rr_features(pre_rr_all, i))
                y.append(aami_mapping[symbol])

    X = np.asarray(X)
    rr = np.asarray(rr)
    y = np.asarray(y)
    print(f'[RECORD] Record {record_name} completed: {len(X)} valid beats extracted (raw, unfiltered).')
    return X, rr, y


def rr_features(pre_rr_all, i):
    """RR_FEATURES of beat i. pre_rr_all[j] is the interval (s) between beat
    j-1 and beat j (pre_rr_all[0] is NaN). The local and long averages use
    only the intervals before beat i; for the first beats of a record, with
    no earlier interval, pre_rr itself is the reference (ratios of 1, log 0)."""
    pre_rr = pre_rr_all[i]
    post_rr = pre_rr_all[i + 1]
    previous = pre_rr_all[1:i]
    local_rr = np.mean(previous[-RR_LOCAL_BEATS:]) if len(previous) else pre_rr
    long_rr = np.mean(previous[-RR_LONG_BEATS:]) if len(previous) else pre_rr
    return tuple(np.log((pre_rr / local_rr, post_rr / local_rr, post_rr / pre_rr, pre_rr / long_rr)))


# 3.1 Local cache of the processed dataset (windows from all records)
def dataset_cache_path(feature, input_size):
    return cache_directory / f'dataset_{feature}_{input_size}.npz'


def cache_is_current(cached):
    """True if an opened dataset cache (np.load) was written by this version
    of the code: same DATASET_CACHE_VERSION and the same records."""
    return ('version' in cached.files
            and int(cached['version']) == DATASET_CACHE_VERSION
            and set(np.unique(cached['records'])) == {int(r) for r in ALL_RECORDS})


def load_cached_dataset(config):
    """Returns (X, rr, y, records) from the raw dataset cache, building it
    first if needed. records holds the MIT-BIH record number of every beat,
    which is what the inter-patient split is based on."""
    cache_path = dataset_cache_path(config.feature, config.input_size)
    if cache_path.exists():
        print(f'[CACHE] Loading cached raw dataset from {cache_path}...')
        with np.load(cache_path) as cached:
            if cache_is_current(cached):
                X_total, rr_total = cached['X'], cached['rr']
                y_total, records = cached['y'], cached['records']
                print(f'[CACHE] Loaded cached dataset: X={X_total.shape}, y={y_total.shape}')
                return X_total, rr_total, y_total, records
        # Caches from an older version (other RR features, no record ids, or
        # a different set of records) are rebuilt from the local MIT-BIH files.
        print('[CACHE] Cached dataset is from an older version, rebuilding it...')

    records_to_load = ALL_RECORDS
    all_X, all_rr, all_y, all_records = [], [], [], []
    print(f"[CONFIG] Number of records: {len(records_to_load)}")

    for record_index, record_name in enumerate(records_to_load, start=1):
        print(f'[DATA] Processing record {record_index}/{len(records_to_load)}: {record_name}')

        # Retry until download/processing succeeds.
        while True:
            try:
                X_patient, rr_patient, y_patient = load_and_segment_record(record_name, config)
                if len(X_patient) > 0:
                    all_X.append(X_patient)
                    all_rr.append(rr_patient)
                    all_y.append(y_patient)
                    all_records.append(np.full(len(X_patient), int(record_name), dtype=np.int16))
                else:
                    print(f'[DATA] Record {record_name} skipped because no valid beats were found.')

                # If no error occurred, move to the next record.
                break

            except Exception as e:
                print(f'[ERROR] Failed to process record {record_name}: {e}')
                print('[RETRY] Retrying in 5 seconds...')
                time.sleep(5)  # Wait 5 seconds before retrying to avoid overloading PhysioNet.

    X_total = np.concatenate(all_X, axis=0).astype(np.float32)
    rr_total = np.concatenate(all_rr, axis=0).astype(np.float32)
    y_total = np.concatenate(all_y, axis=0).astype(np.float32)
    records = np.concatenate(all_records, axis=0)
    print(f'[DATA] Complete dataset shape: X={X_total.shape}, y={y_total.shape}')

    # The cache stores the RAW signal (without filtering or normalization).
    # benchmark_serial.py uses this same cache to build the test set sent to
    # the boards, so the device receives the same signal type as in a real
    # deployment (raw ECG from an ADC) and performs the filtering itself.
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X_total, rr=rr_total, y=y_total, records=records,
                        version=DATASET_CACHE_VERSION)
    print(f'[CACHE] Saved raw dataset to {cache_path}.')
    return X_total, rr_total, y_total, records


def select_records(records, record_names):
    """Boolean mask of the beats that belong to the given records."""
    return np.isin(records, [int(name) for name in record_names])


def print_class_distribution(name, y):
    counts = np.bincount(np.asarray(y).astype(int), minlength=len(classes))
    summary = ', '.join(f'{label}={count}' for label, count in zip(classes, counts))
    print(f'[SPLIT] {name}: {len(y)} beats ({summary})')


# 4. Main loading function
def build_full_dataset(config):
    """With config.split, returns (train, validation, test), split by patient
    (see DS1_RECORDS / TEST_RECORDS above); otherwise returns every beat as a
    single set. Each set has .X (band-pass filtered beat windows), .rr (the
    RR_FEATURES of each beat), .y (class ids) and .records (the MIT-BIH
    record of each beat, for the per-patient report)."""
    print('[START] Dataset loading started.')
    print(f"[CONFIG] Feature: {config.feature}")
    print(f"[CONFIG] Input window size: {config.input_size}")

    X_total, rr_total, y_total, records = load_cached_dataset(config)

    def make_set(name, mask):
        # The filter is applied here only, to the in-memory splits; it is never
        # saved back to the cache. This exactly reproduces what the firmware
        # does with each window received over serial (see filter_beats above).
        print_class_distribution(name, y_total[mask])
        return SimpleNamespace(X=filter_beats(X_total[mask]), rr=rr_total[mask], y=y_total[mask],
                               records=records[mask])

    # Use the config to decide whether to split the dataset.
    if config.split:
        paced = 'with' if INCLUDE_PACED_RECORDS else 'without'
        print(f'[SPLIT] Splitting dataset by patient (inter-patient, Chazal DS1/DS2, '
              f'{paced} paced records)...')
        train = make_set('Training (DS1)', select_records(records, TRAIN_RECORDS))
        validation = make_set('Validation (DS1)', select_records(records, VALIDATION_RECORDS))
        test = make_set('Test (DS2)', select_records(records, TEST_RECORDS))
        print('[DONE] Dataset loading completed.')
        return train, validation, test

    print('[SPLIT] Dataset split disabled.')
    everything = make_set('All records', np.ones(len(y_total), dtype=bool))
    print('[DONE] Dataset loading completed.')
    return everything
