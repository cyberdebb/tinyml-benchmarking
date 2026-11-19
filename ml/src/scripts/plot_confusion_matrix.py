"""Confusion matrices of every model, from the DS2 test run on the boards.

Reads the per-beat files benchmark_serial.py writes (<model>_serial.csv, with
true_class and predicted_class) in <mcu>_firmware/results/inference/ (or
<mcu>_firmware/results/ for older runs) and saves, in ml/figures/:
    confusion_<mcu>_<model>.png/.pdf   one matrix per model and board
    confusion_<mcu>.png/.pdf           all models of a board side by side

Each cell shows the number of beats and, in color, the percentage of the true
class (row): with ~89% of the beats in N, raw counts alone would hide how the
rare classes (S, V, F) are classified.

    python ml/src/scripts/plot_confusion_matrix.py              (both boards)
    python ml/src/scripts/plot_confusion_matrix.py stm32 --format png pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
repository_directory = PROJECT_DIRECTORY.parent
CLASSES = ['N', 'S', 'V', 'F', 'Q']
BOARD_NAMES = {'esp32': 'ESP32-S3', 'stm32': 'NUCLEO-F767ZI'}
MODEL_NAMES = {
    'cnn_resnet': 'ResNet-1D', 'cnn_rn': 'ResNet-1D',
    'cnn_mobilenet_v2': 'MobileNetV2-1D', 'cnn_mn2': 'MobileNetV2-1D',
    'cnn_mobilenet_v3': 'MobileNetV3-1D', 'cnn_mn3': 'MobileNetV3-1D',
    'mlp': 'MLP', 'rf': 'Random Forest', 'svm': 'SVM',
}
MODEL_ORDER = list(MODEL_NAMES)
# One-hue sequential ramp (light -> dark blue); the lightest step stays visible on white.
CMAP = LinearSegmentedColormap.from_list(
    'blues', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
INK, MUTED = '#0b0b0b', '#5f5e5a'


def class_index(value):
    value = value.strip()
    return CLASSES.index(value) if value in CLASSES else int(float(value))


def confusion(path):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            matrix[class_index(row['true_class']), class_index(row['predicted_class'])] += 1
    return matrix


def draw(axes, matrix, title, xlabel=True, ylabel=True):
    totals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0) * 100
    image = axes.imshow(percent, cmap=CMAP, vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            dark = percent[i, j] > 55
            axes.text(j, i - 0.13, f'{matrix[i, j]:,}'.replace(',', '.'), ha='center', va='center',
                      fontsize=8.5, color='white' if dark else INK, fontweight='bold')
            axes.text(j, i + 0.2, f'{percent[i, j]:.1f}%'.replace('.', ','), ha='center',
                      va='center', fontsize=7, color='white' if dark else MUTED)
    axes.set_xticks(range(len(CLASSES)), CLASSES)
    axes.set_yticks(range(len(CLASSES)), [f'{c} ({totals[i, 0]:,})'.replace(',', '.')
                                           for i, c in enumerate(CLASSES)])
    if xlabel:
        axes.set_xlabel('Predicted class', color=INK)
    if ylabel:
        axes.set_ylabel('True class (beats)', color=INK)
    axes.set_title(title, color=INK, fontsize=11)
    axes.tick_params(colors=MUTED, labelcolor=INK, length=0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    return image


def save(figure, destination, formats):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(destination.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(figure)


def find_results(mcu):
    base = repository_directory / f'{mcu}_firmware' / 'results'
    for folder in (base / 'inference', base):
        files = {path.name[:-len('_serial.csv')]: path for path in folder.glob('*_serial.csv')}
        if files:
            order = lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m)
            return [(model, files[model]) for model in sorted(files, key=order)]
    return []


def plot_board(mcu, output, formats):
    results = find_results(mcu)
    if not results:
        print(f'[{mcu}] no <model>_serial.csv found, skipping')
        return
    board = BOARD_NAMES[mcu]
    matrices = []
    for model, path in results:
        matrix = confusion(path)
        matrices.append((model, matrix))
        figure, axes = plt.subplots(figsize=(4.8, 4.2))
        image = draw(axes, matrix, f'{MODEL_NAMES.get(model, model)} — {board}')
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label='% of true class')
        save(figure, output / f'confusion_{mcu}_{model}', formats)
        print(f'[{mcu}] {model}: {matrix.sum():,} beats')

    columns = 3
    rows = -(-len(matrices) // columns)
    figure, axes_grid = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.4 * rows),
                                     squeeze=False, layout='constrained')
    for index, (model, matrix) in enumerate(matrices):
        row, column = divmod(index, columns)
        image = draw(axes_grid[row][column], matrix, MODEL_NAMES.get(model, model),
                     xlabel=index + columns >= len(matrices), ylabel=column == 0)
    for index in range(len(matrices), rows * columns):
        axes_grid[index // columns][index % columns].axis('off')
    figure.colorbar(image, ax=axes_grid, shrink=0.6, label='% of true class')
    save(figure, output / f'confusion_{mcu}', formats)
    print(f'[{mcu}] saved to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mcu', nargs='?', choices=('esp32', 'stm32', 'all'), default='all')
    parser.add_argument('--format', nargs='+', default=['png'], choices=('png', 'pdf', 'svg'))
    parser.add_argument('--output', default=str(PROJECT_DIRECTORY / 'figures'))
    arguments = parser.parse_args()
    for mcu in (('esp32', 'stm32') if arguments.mcu == 'all' else (arguments.mcu,)):
        plot_board(mcu, Path(arguments.output), arguments.format)


if __name__ == '__main__':
    main()
