"""Current/energy test of one model on one board, read from the power logger.

The extra mode next to the DS2 test (benchmark_serial.py). The board under
test runs the energy firmware (<model>_energy environment) and has no serial
link: it loops "idle 5 s -> burst of inferences 3 s" and toggles its sync
pin before each inference. This script only talks to the power logger
(power_logger firmware), which measures every burst with baseline
subtraction and answers one BURST line per burst.

Usage (from the repository root):
    python power_logger/scripts/benchmark_energy.py COM7 esp32 cnn
    python power_logger/scripts/benchmark_energy.py COM7 stm32 rf --bursts 20
    python power_logger/scripts/benchmark_energy.py COM7 esp32 cnn --profile
    python power_logger/scripts/benchmark_energy.py COM7 esp32 cnn --check

COM7 is the logger's port (not the board's). The first burst after the
start is discarded (--warmup): the board may still be booting.

Results go to <mcu>_firmware/results/, next to the DS2 test results:
    <model>_energy.csv           one row per burst (all the BURST fields)
    <model>_energy_report.txt    summary over the bursts
    <model>_current_profile.csv  with --profile: the current of each burst,
                                 averaged over blocks of --block conversions
                                 (Iinst profile; see power_logger/README.md)

Only needs pyserial and numpy.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import serial

repository_directory = Path(__file__).resolve().parents[2]
firmware_directories = {
    'esp32': repository_directory / 'esp32_firmware',
    'stm32': repository_directory / 'stm32_firmware',
}
board_names = {'esp32': 'ESP32-S3', 'stm32': 'STM32 NUCLEO-F767ZI'}
model_names = {'cnn': 'CNN', 'mlp': 'MLP', 'rf': 'Random Forest', 'svm': 'SVM'}

baud_rate = 921600
# The ESP32 ROM prints its boot message at this rate, whatever the firmware.
rom_baud_rate = 115200
ready_timeout_seconds = 15
# One DUT cycle is idle 5 s + burst 3 s; the logger reports a burst ~1 s
# (the end-of-burst gap) after it ends.
burst_timeout_seconds = 30

# Fields summarized in the report: (column, label, format).
summary_fields = (
    ('n_inf', 'Inferences per burst', '.0f'),
    ('t_inf_us', 'Inference time (us)', '.2f'),
    ('E_inf_uJ', 'Einf, energy per inference above idle (uJ)', '.4f'),
    ('E_inf_sigma_uJ', 'Einf noise floor per burst (uJ)', '.4f'),
    ('I_idle_mA', 'Idle current (mA)', '.3f'),
    ('V_idle_V', 'Idle voltage (V)', '.4f'),
    ('P_idle_mW', 'Idle power (mW)', '.3f'),
    ('I_inf_mA', 'Current while inferring, Iinst block mean (mA)', '.3f'),
    ('dI_inf_mA', 'Current above idle while inferring (mA)', '.3f'),
    ('P_inf_mW', 'Power while inferring (mW)', '.3f'),
    ('dP_inf_mW', 'Power above idle while inferring (mW)', '.3f'),
    ('E_net_uJ', 'Energy above idle per burst (uJ)', '.2f'),
)


def to_number(text):
    try:
        return float(text)
    except ValueError:
        return math.nan


def reset_logger(connection):
    """Resets the logger ESP32 through the auto-reset circuit (EN on RTS).
    Its 3V3 regulator stays on, so the board under test keeps running."""
    connection.dtr = False
    connection.rts = True
    time.sleep(0.1)
    connection.rts = False
    time.sleep(0.5)


def read_line(connection):
    """One line from the logger, with the bytes that aren't printable ASCII
    removed. Returns None for a line of garbage: after a reset the ESP32 ROM
    prints its boot message at 115200 baud, which reads as noise at 921600."""
    raw = connection.readline()
    if not raw:
        return ''
    text = raw.decode('ascii', errors='replace')
    printable = ''.join(character for character in text if ' ' <= character <= '~')
    if len(printable) < 0.8 * len(text.strip()):
        return None
    return printable.strip()


def diagnose_at_rom_baud(connection):
    """Resets the logger again and reads it at 115200 baud, where the ESP32
    ROM boot message (and a firmware left at the default baud rate) can be
    read, then says what that means."""
    print(f'[DIAG] Resetting the logger and reading it at {rom_baud_rate} baud...')
    connection.baudrate = rom_baud_rate
    connection.reset_input_buffer()
    reset_logger(connection)
    lines = []
    deadline = time.time() + 6
    next_info = time.time() + 2
    while time.time() < deadline:
        if time.time() > next_info:
            connection.write(b'INFO\n')
            connection.flush()
            next_info = time.time() + 2
        line = read_line(connection)
        if line:
            lines.append(line)
            print(f'  [logger @ {rom_baud_rate}] {line}')
    text = '\n'.join(lines)

    if 'BROWNOUT' in text.upper() or text.count('rst:') > 2:
        print('[DIAG] The logger keeps resetting (brownout): its 3V3 drops, most likely when '
              'the board under test starts drawing current through the INA226. Use a '
              'shorter/better USB cable or a USB port that gives more current, and check '
              'that the board under test is not ALSO on USB (two 3V3 sources).')
    elif 'waiting for download' in text:
        print('[DIAG] The logger is stuck in download mode (GPIO0 low at reset). Unplug it, '
              'make sure nothing holds GPIO0/BOOT, and plug it back.')
    elif 'COLUMNS,' in text or 'READY' in text or text.startswith('INFO,') or '\nINFO,' in text:
        print(f'[DIAG] The power logger firmware answers at {rom_baud_rate} baud, not 921600: '
              'it was built with the old console setting. Run again with '
              f'--baud {rom_baud_rate}, or reflash it (cd power_logger, pio run -t upload) '
              'after checking CONFIG_ESP_CONSOLE_UART_BAUDRATE=921600 in sdkconfig.esp32dev.')
    elif 'app_main' in text or 'main_task' in text or 'cpu_start' in text:
        print('[DIAG] The ESP32 boots a program that is not the power logger (probably the '
              'empty project). Flash it: cd power_logger, pio run -t upload.')
    elif lines:
        print('[DIAG] The logger boots but does not answer as the power logger. Flash it '
              '(cd power_logger, pio run -t upload) and send the lines above if it persists.')
    else:
        print('[DIAG] Nothing readable at 115200 either: check that COM port is the ESP32 '
              'extra (the logger), not the board under test.')


def parse_columns(line, columns):
    if line.startswith('COLUMNS,'):
        parts = line.split(',')
        columns[parts[1]] = parts[2:]


def wait_for_ready(connection):
    """Waits for the logger after its reset; returns {kind: [field names]}
    from its COLUMNS lines.

    Ready is the READY line at the end of the boot, or -- if the boot log
    was missed or garbled -- the answer to INFO, which repeats the COLUMNS
    lines. INFO is sent every few seconds until the logger answers."""
    deadline = time.time() + ready_timeout_seconds
    next_info = time.time() + 3
    columns = {}
    garbage = 0
    readable = 0
    while time.time() < deadline:
        if time.time() > next_info:
            connection.write(b'INFO\n')
            connection.flush()
            next_info = time.time() + 3
        line = read_line(connection)
        if line is None:
            garbage += 1
            if garbage == 1:
                print('  [logger] (unreadable bytes: the ESP32 ROM boot message at 115200 '
                      'baud, normal after a reset)')
            continue
        if not line:
            continue
        readable += 1
        print(f'  [logger] {line}')
        parse_columns(line, columns)
        if line == 'READY' or (line.startswith('COLUMNS,I,') and 'BURST' in columns):
            return columns
        if line.startswith('COLUMNS,I,'):
            print('[ERROR] The logger firmware has no AUTO mode (no BURST columns). '
                  'Flash the current power_logger firmware.')
            sys.exit(1)

    print('[ERROR] Timed out waiting for the logger.')
    if readable == 0 and garbage > 0:
        print(f'[ERROR] Only unreadable bytes came at {connection.baudrate} baud.')
        if connection.baudrate != rom_baud_rate:
            diagnose_at_rom_baud(connection)
    elif readable == 0:
        print('[ERROR] Nothing came: check the port (it is the logger ESP32, not the board '
              'under test) and that no serial monitor has it open.')
    else:
        print('[ERROR] The logger answered but never got ready: check the INA226 wiring '
              '(it retries every 1 s until it finds the INA226).')
    sys.exit(1)


def command(connection, text, expect_ok=True):
    """Sends one command; returns the answer lines up to OK/ERR."""
    connection.write((text + '\n').encode('ascii'))
    connection.flush()
    deadline = time.time() + 5
    answer = []
    while time.time() < deadline:
        line = read_line(connection)
        if not line:
            continue
        answer.append(line)
        name = text.split()[0].upper()
        if line == f'OK,{name}' or line.startswith(('ERR,', 'PONG', 'READ,', 'INFO,')):
            if expect_ok and line.startswith('ERR,'):
                print(f'[ERROR] {text}: {line}')
                sys.exit(1)
            return answer
    print(f'[ERROR] No answer from the logger to {text}')
    sys.exit(1)


def check_setup(connection):
    """Prints the logger readings for a few seconds: current, voltage and
    sync edges, to check the wiring before a test."""
    print('\n[CHECK] 10 readings, 1 s apart. The current should be the board\'s '
          '(tens of mA), the voltage ~3.3 V, and sync_edges should jump every '
          '~8 s while the energy firmware bursts.')
    print('        I_mA      V_V     P_mW  sync_edges  i2c_errors')
    for _ in range(10):
        for line in command(connection, 'READ', expect_ok=False):
            if line.startswith('READ,'):
                fields = line.split(',')[1:]
                print(f'  {float(fields[0]):9.3f} {float(fields[1]):8.4f} '
                      f'{float(fields[2]):8.3f} {int(fields[7]):11d} {int(fields[6]):11d}')
        time.sleep(1)


def run_bursts(connection, columns, arguments, profile_writer):
    """Sends AUTO and collects the BURST lines; returns them as dicts
    (warm-up bursts excluded)."""
    total = arguments.warmup + arguments.bursts
    burst_fields = columns['BURST']
    command(connection, f'AUTO {total} {arguments.gap}')
    print(f'\n[AUTO] Measuring {arguments.bursts} bursts (+{arguments.warmup} warm-up). '
          f'One burst every ~8 s, so ~{total * 8 // 60 + 1} min.')

    bursts = []
    profile_rows = {}  # window -> rows waiting for its BURST line (idle current)
    deadline = time.time() + burst_timeout_seconds
    while True:
        if time.time() > deadline:
            print('\n[ERROR] No burst from the board in '
                  f'{burst_timeout_seconds} s. Check that:')
            print('  - the board runs the energy firmware (pio run -e <model>_energy -t upload);')
            print('  - the sync wire goes from the board (ESP32-S3 GPIO4 / STM32 PF13) to the '
                  'logger GPIO4, with GND shared;')
            print('  - the board is powered through the INA226 (run with --check).')
            raise TimeoutError('no burst')
        line = read_line(connection)
        if not line:
            continue
        if line.startswith('I,'):
            if profile_writer is not None:
                window, t_us, current, power = line.split(',')[1:]
                profile_rows.setdefault(int(window), []).append(
                    (int(t_us), to_number(current), to_number(power)))
            continue
        if line.startswith('BURST,'):
            values = dict(zip(burst_fields, line.split(',')[1:]))
            burst = {key: to_number(value) for key, value in values.items()}
            index = int(burst['index'])
            warmup = index <= arguments.warmup
            rows = profile_rows.pop(index, [])
            label = 'warm-up, discarded' if warmup else f'{index - arguments.warmup}/{arguments.bursts}'
            print(f"  burst {label}: {burst['n_inf']:.0f} inferences of "
                  f"{burst['t_inf_us']:.1f} us, Einf {burst['E_inf_uJ']:.4f} uJ, "
                  f"I {burst['I_idle_mA']:.2f} -> {burst['I_inf_mA']:.2f} mA")
            if not warmup:
                bursts.append(burst)
                if profile_writer is not None:
                    idle = burst['I_idle_mA']
                    for t_us, current, power in rows:
                        profile_writer.writerow([index - arguments.warmup, t_us,
                                                 f'{current:.3f}', f'{current - idle:.3f}',
                                                 f'{power:.3f}'])
            deadline = time.time() + burst_timeout_seconds
            continue
        if line.startswith('AUTO_DONE'):
            return bursts
        if line.startswith('#') and 'Burst' in line:
            continue  # the human summary of the BURST line just printed
        print(f'  [logger] {line}')


def statistics(values):
    values = np.array([value for value in values if math.isfinite(value)])
    if len(values) == 0:
        return None
    return (values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0,
            values.min(), values.max(), len(values))


def ds2_inference_us(firmware_directory, model):
    """Mean inference time of the DS2 test (benchmark_serial.py), if run."""
    serial_file = firmware_directory / 'results' / f'{model}_serial.csv'
    if not serial_file.exists():
        return None
    with open(serial_file, newline='', encoding='utf-8') as handle:
        times = [float(row['inference_us']) for row in csv.DictReader(handle)]
    return float(np.mean(times)) if times else None


def report_text(arguments, bursts, discarded, ds2_us):
    lines = [
        '==================================================',
        f'  Current/energy test: {model_names[arguments.model]} on the '
        f'{board_names[arguments.mcu]}',
        f'  Bursts: {len(bursts)} used'
        + (f', {discarded} discarded (no idle baseline or < 2 inferences)' if discarded else ''),
        '  Einf = (E_burst - P_idle * T) / N  (baseline subtraction, idle right before each burst)',
        '  Values: mean +- std over the bursts [min .. max]',
        '',
    ]
    for key, label, spec in summary_fields:
        stats = statistics(burst[key] for burst in bursts)
        if stats is None:
            lines.append(f'  {label}: nan')
            continue
        mean, std, low, high, _ = stats
        lines.append(f'  {label}: {mean:{spec}} +- {std:{spec}} [{low:{spec}} .. {high:{spec}}]')

    energy = statistics(burst['E_inf_uJ'] for burst in bursts)
    if energy is not None and energy[4] > 1:
        standard_error = energy[1] / math.sqrt(energy[4])
        lines.append(f'  Einf standard error of the mean: {standard_error:.4f} uJ '
                     f'({100 * standard_error / abs(energy[0]) if energy[0] else math.nan:.2f}%)')
    timing = statistics(burst['t_inf_us'] for burst in bursts)
    if ds2_us is not None and timing is not None:
        lines.append(f'  DS2 test inference time (benchmark_serial.py): {ds2_us:.2f} us '
                     f'(this test: {timing[0]:.2f} us, from the sync edges)')
    missed = sum(burst['missed'] for burst in bursts if math.isfinite(burst['missed']))
    dropped = sum(burst['stream_dropped'] for burst in bursts
                  if math.isfinite(burst['stream_dropped']))
    if missed:
        lines.append(f'  Note: {missed:.0f} INA226 conversions missed by the logger '
                     '(already accounted for in the time weighting)')
    if dropped:
        lines.append(f'  Note: {dropped:.0f} profile points dropped (use a larger --block)')
    lines.append('==================================================')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Current/energy test of one model on one board, via the power logger.')
    parser.add_argument('port', help='Serial port of the power logger ESP32, e.g. COM7 or /dev/ttyUSB1')
    parser.add_argument('mcu', choices=tuple(firmware_directories),
                        help='Board under test (where the results are saved)')
    parser.add_argument('model', choices=tuple(model_names),
                        help='Model in the energy firmware flashed on the board')
    parser.add_argument('--bursts', type=int, default=10, help='Bursts to measure (default 10)')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Bursts discarded at the start (default 1)')
    parser.add_argument('--gap', type=int, default=1000,
                        help='End-of-burst gap in ms: longer than one inference (default 1000)')
    parser.add_argument('--profile', action='store_true',
                        help='Also save the current profile (Iinst) of every burst')
    parser.add_argument('--block', type=int, default=1,
                        help='INA226 conversions averaged per profile point (default 1, ~0.34 ms)')
    parser.add_argument('--check', action='store_true',
                        help='Only print a few logger readings to check the wiring')
    parser.add_argument('--baud', type=int, default=baud_rate)
    arguments = parser.parse_args()
    if arguments.bursts < 1 or arguments.warmup < 0 or arguments.block < 1:
        parser.error('--bursts >= 1, --warmup >= 0 and --block >= 1')

    firmware_directory = firmware_directories[arguments.mcu]
    results_directory = firmware_directory / 'results'
    energy_file = results_directory / f'{arguments.model}_energy.csv'
    report_file = results_directory / f'{arguments.model}_energy_report.txt'
    profile_file = results_directory / f'{arguments.model}_current_profile.csv'

    with serial.Serial(arguments.port, arguments.baud, timeout=1) as connection:
        reset_logger(connection)
        print('[SERIAL] Waiting for the power logger...')
        columns = wait_for_ready(connection)

        if arguments.check:
            check_setup(connection)
            return

        command(connection, f'BLOCK {arguments.block}')
        command(connection, f'STREAM {1 if arguments.profile else 0}')

        results_directory.mkdir(parents=True, exist_ok=True)
        profile_handle = None
        profile_writer = None
        if arguments.profile:
            profile_handle = open(profile_file, 'w', newline='', encoding='utf-8')
            profile_writer = csv.writer(profile_handle)
            profile_writer.writerow(['burst', 't_us', 'I_mA', 'dI_mA', 'P_mW'])
        try:
            bursts = run_bursts(connection, columns, arguments, profile_writer)
        except (TimeoutError, KeyboardInterrupt, serial.SerialException) as error:
            try:
                connection.write(b'ABORT\n')
            except serial.SerialException:
                pass
            print(f'\n[STOPPED] {type(error).__name__}: {error}. Nothing saved.')
            sys.exit(1)
        finally:
            if profile_handle is not None:
                profile_handle.close()

    valid = [burst for burst in bursts
             if math.isfinite(burst['E_inf_uJ']) and burst['n_inf'] >= 2]
    if not valid:
        print('[ERROR] No valid burst (no idle baseline or no inferences counted).')
        sys.exit(1)

    with open(energy_file, 'w', newline='', encoding='utf-8') as handle:
        fields = ['burst'] + [key for key in columns['BURST'] if key != 'index']
        writer = csv.writer(handle)
        writer.writerow(['mcu', 'model'] + fields)
        for number, burst in enumerate(bursts, start=1):
            writer.writerow([arguments.mcu, arguments.model, number]
                            + [burst[key] for key in fields[1:]])

    text = report_text(arguments, valid, len(bursts) - len(valid),
                       ds2_inference_us(firmware_directory, arguments.model))
    print('\n' + text)
    report_file.write_text(text + '\n', encoding='utf-8')
    saved = [energy_file, report_file] + ([profile_file] if arguments.profile else [])
    print('\n[DONE] Results saved to ' + ', '.join(str(path) for path in saved))


if __name__ == '__main__':
    main()
