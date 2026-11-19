"""Figure with one example of each MIT-BIH beat type, grouped by AAMI class.

For each annotation symbol of the AAMI mapping (load_data.aami_mapping), all
the beats of the MIT-BIH records are collected (MLII lead, 256-sample window
centered on the annotated R peak, the same band-pass filter as the models) and
the most typical one is shown: the beat closest to the median waveform of that
symbol. The paced records (107 and 217, the ones with MLII) are included so
the paced beats ('/', 'f') appear in the figure, even though they are left out
of training and test.

Run from the repository root (records are downloaded on first use, like in
training):
    python ml/src/scripts/plot_beat_examples.py
Saves ml/figures/beat_examples.png
"""

import argparse
from pathlib import Path
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIRECTORY / 'src'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wfdb  # noqa: E402

import load_data  # noqa: E402
from load_data import aami_mapping, classes, ensure_record_local, filter_beats, sampling_rate  # noqa: E402

WINDOW = 256
MAX_BEATS_PER_SYMBOL = 2000   # enough to find a typical beat
SYMBOL_NAMES = {
    'N': 'Normal', 'L': 'Left bundle branch block', 'R': 'Right bundle branch block',
    'e': 'Atrial escape', 'j': 'Junctional escape',
    'A': 'Atrial premature', 'a': 'Aberrant atrial premature',
    'J': 'Junctional premature', 'S': 'Supraventricular premature',
    'V': 'Premature ventricular', 'E': 'Ventricular escape',
    'F': 'Ventricular and normal fusion',
    '/': 'Paced', 'f': 'Paced and normal fusion', 'Q': 'Unclassifiable',
}
CLASS_NAMES = {'N': 'N (normal)', 'S': 'S (supraventricular)', 'V': 'V (ventricular)',
               'F': 'F (fusion)', 'Q': 'Q (unknown)'}
CLASS_COLORS = {'N': '#2a78d6', 'S': '#eb6834', 'V': '#1baf7a', 'F': '#c98500', 'Q': '#e87ba4'}
INK, MUTED, GRID = '#0b0b0b', '#5f5e5a', '#e4e3df'


def collect_beats(records, data_directory):
    load_data.data_directory = Path(data_directory)
    beats = {symbol: [] for symbol in aami_mapping}
    counts = {symbol: 0 for symbol in aami_mapping}
    half = WINDOW // 2
    for record_name in records:
        ensure_record_local(record_name)
        path = str(load_data.data_directory / record_name)
        record = wfdb.rdrecord(path)
        if 'MLII' not in record.sig_name:
            print(f'[SKIP] {record_name}: no MLII lead')
            continue
        signal = record.p_signal[:, record.sig_name.index('MLII')]
        annotation = wfdb.rdann(path, 'atr')
        for position, symbol in zip(annotation.sample, annotation.symbol):
            if symbol not in beats or not (half <= position <= len(signal) - half):
                continue
            counts[symbol] += 1
            if len(beats[symbol]) < MAX_BEATS_PER_SYMBOL:
                beats[symbol].append(signal[position - half:position + half])
        print(f'[RECORD] {record_name} read')
    return beats, counts


def typical_beat(windows):
    """The filtered beat closest (Euclidean) to the median waveform."""
    filtered = filter_beats(np.asarray(windows))
    median = np.median(filtered, axis=0)
    return filtered[np.argmin(np.linalg.norm(filtered - median, axis=1))]


def plot(beats, counts, destination):
    by_class = {name: [s for s, c in aami_mapping.items() if classes[c] == name] for name in classes}
    columns = max(len(symbols) for symbols in by_class.values())
    figure, axes = plt.subplots(len(classes), columns, figsize=(3.0 * columns, 2.3 * len(classes)),
                                sharex=True, squeeze=False)
    t_ms = (np.arange(WINDOW) - WINDOW // 2) / sampling_rate * 1000
    for row, name in enumerate(classes):
        for column in range(columns):
            axes_ = axes[row][column]
            symbols = by_class[name]
            if column >= len(symbols) or not beats[symbols[column]]:
                axes_.axis('off')
                continue
            symbol = symbols[column]
            axes_.plot(t_ms, typical_beat(beats[symbol]), color=CLASS_COLORS[name], lw=1.6)
            axes_.axvline(0, color=GRID, lw=1, zorder=0)
            axes_.set_title(f"{SYMBOL_NAMES[symbol]} ('{symbol}')\n{counts[symbol]:,} beats"
                            .replace(',', '.'), fontsize=9, color=INK)
            axes_.tick_params(labelsize=8, colors=MUTED, labelcolor=INK)
            for side in ('top', 'right'):
                axes_.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                axes_.spines[side].set_color(MUTED)
            if column == 0:
                axes_.set_ylabel(f'{CLASS_NAMES[name]}\n\nmV', fontsize=9.5, color=INK)
            if row == len(classes) - 1 or column >= len(by_class[classes[row + 1]]):
                axes_.set_xlabel('time relative to R peak (ms)', fontsize=8.5, color=INK)
                axes_.tick_params(labelbottom=True)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_path = destination if destination.suffix.lower() == '.png' else destination.with_suffix('.png')
    figure.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'[DONE] Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=str(load_data.data_directory),
                        help='MIT-BIH folder (default: the one used in training)')
    parser.add_argument('--output', default=str(PROJECT_DIRECTORY / 'figures' / 'beat_examples'))
    arguments = parser.parse_args()
    beats, counts = collect_beats(load_data.ALL_RECORDS, arguments.data_dir)
    for symbol, count in counts.items():
        print(f'  {symbol}: {count}')
    plot(beats, counts, Path(arguments.output))


if __name__ == '__main__':
    main()