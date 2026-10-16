"""Plots of the current/energy test results (benchmark_energy.py).

Reads, for each MCU, the files benchmark_energy.py writes in
<mcu>_firmware/results/energy/:
    <model>_energy.csv           one row per burst
    <model>_current_profile.csv  Iinst profile (only with --profile)

and saves, in power_logger/results/<mcu>/:
    <model>_current_voltage.png    current and voltage over time, one burst
                                   (two panels sharing the time axis)
    <model>_voltage_x_current.png  voltage vs current of every profile point,
                                   with a linear fit (slope = -R of the supply
                                   path: logger 3V3 regulator, wires, shunt)
    summary_energy.png             Einf (log scale) and power above idle of
                                   every model, mean +- std over the bursts

and, when both MCUs have results, in power_logger/results/:
    comparison_energy.png          Einf of every model on both boards side by
                                   side (log scale), with the Nucleo/ESP32-S3
                                   ratio next to each model

The profile has no voltage column: V = P / I, both from the same INA226
reading. Each profile point is the block mean of --block INA226 conversions
(~0.34 ms each), so inferences faster than that (RF, MLP) show as a plateau.

    python scripts/plot_energy.py (both MCUs)
    python scripts/plot_energy.py stm32 --burst 3
    python scripts/plot_energy.py esp32 --models cnn_resnet mlp --format png pdf

Needs numpy and matplotlib.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

repository_directory = Path(__file__).resolve().parents[2]
firmware_directories = {
    'esp32': repository_directory / 'esp32_firmware',
    'stm32': repository_directory / 'stm32_firmware',
}
board_names = {'esp32': 'ESP32-S3', 'stm32': 'STM32 NUCLEO-F767ZI'}
# Display names and order; unknown models are kept with their file name.
model_names = {
    'cnn_resnet': 'ResNet-1D',
    'cnn_rn': 'ResNet-1D',
    'cnn_mobilenet_v2': 'MobileNetV2-1D',
    'cnn_mn2': 'MobileNetV2-1D',
    'cnn_mobilenet_v3': 'MobileNetV3-1D',
    'cnn_mn3': 'MobileNetV3-1D',
    'mlp': 'MLP',
    'rf': 'Random Forest',
    'svm': 'SVM',
}
model_order = list(model_names)

# Colors: one series per chart, so one hue; ink and grid stay neutral.
series_color = '#2a78d6'
reference_color = '#eb6834'  # idle / mean reference lines
ink = '#0b0b0b'
muted_ink = '#5f5e5a'
grid_color = '#e4e3df'
# Comparison plot: one fixed hue per board (categorical slots 1 and 2).
board_colors = {'esp32': '#2a78d6', 'stm32': '#eb6834'}


def decimal_comma(text):
    return text.replace(',', '\u2009').replace('.', ',')


def energy_label(value):
    """29 442 / 785 / 44,9: no scientific notation, decimal comma."""
    return decimal_comma(f'{value:,.0f}' if value >= 100 else f'{value:.3g}')


def ratio_label(value):
    return decimal_comma(f'{value:.1f}') + '×'


def style_axes(axes):
    axes.grid(True, color=grid_color, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(muted_ink)
    axes.tick_params(colors=muted_ink, labelcolor=ink)


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def column(rows, name):
    return np.array([float(row[name]) for row in rows])


def model_label(model):
    return model_names.get(model, model)


def save(figure, destination, formats):
    for extension in formats:
        figure.savefig(destination.with_suffix('.' + extension), dpi=200, bbox_inches='tight')
    plt.close(figure)


def profile_arrays(profile_rows, burst):
    rows = [row for row in profile_rows if int(row['burst']) == burst]
    if not rows:
        return None
    t_s = column(rows, 't_us') / 1e6
    current = column(rows, 'I_mA')
    power = column(rows, 'P_mW')
    with np.errstate(divide='ignore', invalid='ignore'):
        voltage = np.where(current > 1.0, power / current, np.nan)
    return t_s, current, voltage


def plot_time_series(profile_rows, energy_row, title, burst, destination, formats):
    arrays = profile_arrays(profile_rows, burst)
    if arrays is None:
        print(f'  [WARNING] burst {burst} not in the profile, skipping the time plot')
        return
    t_s, current, voltage = arrays
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                         gridspec_kw={'hspace': 0.12})
    top.plot(t_s, current, color=series_color, linewidth=1.2)
    idle = float(energy_row['I_idle_mA'])
    inferring = float(energy_row['I_inf_mA'])
    top.axhline(idle, color=reference_color, linewidth=1, linestyle='--')
    top.axhline(inferring, color=muted_ink, linewidth=1, linestyle=':')
    top.annotate(f'idle {idle:.1f} mA', (t_s[-1], idle), xytext=(0, 4),
                 textcoords='offset points', ha='right', va='bottom', color=ink, fontsize=8)
    top.annotate(f'inferência {inferring:.1f} mA', (t_s[-1], inferring), xytext=(0, 4),
                 textcoords='offset points', ha='right', va='bottom', color=ink, fontsize=8)
    top.set_ylabel('Corrente (mA)', color=ink)
    top.set_title(title, color=ink, loc='left', fontsize=11)

    bottom.plot(t_s, voltage, color=series_color, linewidth=1.2)
    bottom.set_ylabel('Tensão (V)', color=ink)
    bottom.set_xlabel('Tempo (s)', color=ink)
    for axes in (top, bottom):
        style_axes(axes)
    save(figure, destination, formats)


def plot_voltage_vs_current(profile_rows, title, destination, formats):
    current = column(profile_rows, 'I_mA')
    power = column(profile_rows, 'P_mW')
    keep = current > 1.0
    current, voltage = current[keep], power[keep] / current[keep]
    figure, axes = plt.subplots(figsize=(6, 4.5))
    axes.scatter(current, voltage, s=8, color=series_color, alpha=0.25, linewidths=0)
    if np.ptp(current) > 1.0:
        slope, intercept = np.polyfit(current, voltage, 1)  # V per mA
        line = np.array([current.min(), current.max()])
        axes.plot(line, slope * line + intercept, color=reference_color, linewidth=1.5)
        axes.annotate(f'ajuste linear: V = {intercept:.3f} V − {-slope * 1000:.2f} Ω × I',
                      (0.02, 0.04), xycoords='axes fraction', color=ink, fontsize=8)
    axes.set_xlabel('Corrente (mA)', color=ink)
    axes.set_ylabel('Tensão (V)', color=ink)
    axes.set_title(title, color=ink, loc='left', fontsize=11)
    style_axes(axes)
    save(figure, destination, formats)


def plot_summary(summaries, title, destination, formats):
    labels = [model_label(model) for model, _ in summaries]
    positions = np.arange(len(summaries))
    panels = (
        ('E_inf_uJ', 'Energia por inferência acima do idle (µJ)', True, energy_label),
        ('dP_inf_mW', 'Potência acima do idle (mW)', False,
         lambda value: decimal_comma(f'{value:.1f}')),
    )
    figure, axes_list = plt.subplots(1, 2, figsize=(11, 0.6 * len(summaries) + 1.8))
    for axes, (name, xlabel, log_scale, fmt) in zip(axes_list, panels):
        means = np.array([rows[name].mean() for _, rows in summaries])
        stds = np.array([rows[name].std(ddof=1) if len(rows[name]) > 1 else 0.0
                         for _, rows in summaries])
        axes.barh(positions, means, height=0.6, color=series_color, xerr=stds,
                  error_kw={'ecolor': muted_ink, 'elinewidth': 1, 'capsize': 2})
        for position, mean in zip(positions, means):
            axes.annotate(fmt(mean), (mean, position), xytext=(4, 0),
                          textcoords='offset points', va='center', color=ink, fontsize=8)
        if log_scale:
            axes.set_xscale('log')
        axes.set_yticks(positions, labels)
        axes.invert_yaxis()
        axes.set_xlabel(xlabel, color=ink)
        axes.margins(x=0.15)
        style_axes(axes)
        axes.grid(False, axis='y')
    axes_list[1].set_yticklabels([])
    figure.suptitle(title, color=ink, x=0.01, ha='left', fontsize=11)
    save(figure, destination, formats)


def plot_comparison(per_board, destination, formats):
    """Einf of each model on both boards. per_board: {mcu: {label: E_inf array}}."""
    boards = [mcu for mcu in ('esp32', 'stm32') if per_board.get(mcu)]
    labels = [label for label in dict.fromkeys(model_label(m) for m in model_order)
              if all(label in per_board[mcu] for mcu in boards)]
    if len(boards) < 2 or not labels:
        print('[comparison] needs the same models on both boards, skipping')
        return
    positions = np.arange(len(labels))
    height = 0.38
    figure, axes = plt.subplots(figsize=(8, 0.75 * len(labels) + 1.6))
    for offset, mcu in zip((-height / 2 - 0.01, height / 2 + 0.01), boards):
        values = [per_board[mcu][label] for label in labels]
        means = np.array([v.mean() for v in values])
        stds = np.array([v.std(ddof=1) if len(v) > 1 else 0.0 for v in values])
        axes.barh(positions + offset, means, height=height, color=board_colors[mcu],
                  xerr=stds, label=board_names[mcu],
                  error_kw={'ecolor': muted_ink, 'elinewidth': 1, 'capsize': 2})
        for position, mean in zip(positions + offset, means):
            axes.annotate(energy_label(mean), (mean, position), xytext=(4, 0),
                          textcoords='offset points', va='center', color=ink, fontsize=8)
    # Nucleo / ESP32-S3 ratio, in a column right of the plot.
    for position, label in zip(positions, labels):
        ratio = per_board['stm32'][label].mean() / per_board['esp32'][label].mean()
        axes.annotate(ratio_label(ratio), (1.02, position), xycoords=('axes fraction', 'data'),
                      va='center', ha='left', color=ink, fontsize=9)
    axes.annotate('Nucleo /\nESP32-S3', (1.02, -0.75), xycoords=('axes fraction', 'data'),
                  va='bottom', ha='left', color=muted_ink, fontsize=8)
    axes.set_xscale('log')
    axes.set_yticks(positions, labels)
    axes.invert_yaxis()
    axes.set_xlabel('Energia por inferência acima do idle (µJ)', color=ink)
    axes.margins(x=0.12)
    style_axes(axes)
    axes.grid(False, axis='y')
    axes.legend(loc='lower right', frameon=False, labelcolor=ink, fontsize=9)
    save(figure, destination, formats)


def find_models(results_directory, requested):
    found = sorted(path.name[:-len('_energy.csv')]
                   for path in results_directory.glob('*_energy.csv'))
    if requested:
        found = [model for model in found if model in requested]
    return sorted(found, key=lambda m: (model_order.index(m) if m in model_order
                                        else len(model_order), m))


def plot_mcu(mcu, arguments):
    results_directory = (Path(arguments.input) if arguments.input
                         else firmware_directories[mcu] / 'results' / 'energy')
    output_directory = Path(arguments.output) / mcu
    models = find_models(results_directory, arguments.models)
    if not models:
        print(f'[{mcu}] no <model>_energy.csv in {results_directory}, skipping')
        return {}
    output_directory.mkdir(parents=True, exist_ok=True)
    board = board_names[mcu]
    summaries = []
    for model in models:
        energy_rows = read_rows(results_directory / f'{model}_energy.csv')
        numeric = {name: column(energy_rows, name) for name in ('E_inf_uJ', 'dP_inf_mW')}
        summaries.append((model, numeric))
        profile_file = results_directory / f'{model}_current_profile.csv'
        if not profile_file.exists():
            print(f'[{mcu}] {model}: no profile (run benchmark_energy.py with --profile)')
            continue
        profile_rows = read_rows(profile_file)
        energy_row = next((row for row in energy_rows
                           if int(row['burst']) == arguments.burst), energy_rows[0])
        title = f'{model_label(model)} — {board}'
        plot_time_series(profile_rows, energy_row, f'{title} (burst {arguments.burst})',
                         arguments.burst, output_directory / f'{model}_current_voltage',
                         arguments.format)
        plot_voltage_vs_current(profile_rows, f'{title}: tensão x corrente',
                                output_directory / f'{model}_voltage_x_current',
                                arguments.format)
        print(f'[{mcu}] {model}: plots saved')
    plot_summary(summaries, f'{board}: energia e potência por modelo (média ± desvio)',
                 output_directory / 'summary_energy', arguments.format)
    print(f'[{mcu}] saved to {output_directory}')
    return {model_label(model): numeric['E_inf_uJ'] for model, numeric in summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mcu', nargs='?', choices=('esp32', 'stm32', 'all'), default='all')
    parser.add_argument('--models', nargs='+', help='Only these models (file name prefix)')
    parser.add_argument('--burst', type=int, default=1,
                        help='Burst shown in the time plot (default 1)')
    parser.add_argument('--format', nargs='+', default=['png'], choices=('png', 'pdf', 'svg'))
    parser.add_argument('--input', help='Results folder to read (default <mcu>_firmware/results/energy)')
    parser.add_argument('--output', default=str(repository_directory / 'power_logger' / 'results'),
                        help='Output folder (default power_logger/results)')
    arguments = parser.parse_args()
    per_board = {}
    for mcu in (('esp32', 'stm32') if arguments.mcu == 'all' else (arguments.mcu,)):
        per_board[mcu] = plot_mcu(mcu, arguments)
    if arguments.mcu == 'all':
        plot_comparison(per_board, Path(arguments.output) / 'comparison_energy',
                        arguments.format)


if __name__ == '__main__':
    sys.exit(main())
