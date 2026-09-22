import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

project_directory = Path(__file__).resolve().parent
src_directory = project_directory / 'src'
classifiers_directory = src_directory / 'classifiers'
models_directory = project_directory / 'models'
firmware_directory = project_directory.parent / 'esp32_firmware'
firmware_src_directory = firmware_directory / 'src'

# Maps each model name to its training script
classifier_scripts = {
    'cnn': 'cnn_classifier.py',
    'mlp': 'mlp_classifier.py',
    'rf': 'random_forest_classifier.py',
    'svm': 'svm_classifier.py',
}

tflite_files = ('cnn_classifier.tflite', 'mlp_classifier.tflite')
header_files = (
    'svm_classifier.h',
    'svm_classifier_scaler.h',
    'random_forest_classifier.h',
)


# ---------------------------------------------------------------------------
# Step 1: training
# ---------------------------------------------------------------------------

def run_classifier(classifier, environment):
    """Runs a single classifier script.

    No retries: the dataset is cached on disk before training starts, so a
    failure here is a real error and running the same training again would
    only waste time.
    """
    classifier_path = classifiers_directory / classifier
    print(f'\n[RUN] Starting {classifier}...')

    # -u keeps the child output unbuffered so it shows up in real time.
    process = subprocess.Popen(
        [sys.executable, '-u', str(classifier_path)],
        cwd=project_directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    try:
        for line in process.stdout:
            print(line, end='')
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        raise

    if returncode == 0:
        print(f'[DONE] {classifier} completed successfully.')
        return True

    print(f'[ERROR] {classifier} failed with exit code {returncode}.')
    return False


def run_trainings(models):
    """Step 1: Train the given ML classifiers sequentially.

    Args:
        models: List of model names to train, in any order. Valid names are
                'cnn', 'mlp', 'rf' and 'svm'. A single name may also be passed
                as a plain string.
    """
    if isinstance(models, str):
        models = [models]

    # Normalize and drop duplicates while keeping the order the caller asked for.
    requested = list(dict.fromkeys(model.strip().lower() for model in models))

    if not requested:
        print('[ERROR] No model given. Choose one or more of: '
              f"{', '.join(classifier_scripts)}.")
        sys.exit(1)

    invalid = [model for model in requested if model not in classifier_scripts]
    if invalid:
        print(f"[ERROR] Invalid model(s): {', '.join(invalid)}. "
              f"Choose one or more of: {', '.join(classifier_scripts)}.")
        sys.exit(1)

    classifiers = [classifier_scripts[model] for model in requested]
    print(f"[PIPELINE] Step 1: Running ML training for: {', '.join(requested)}")

    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(src_directory)

    # Keep going even if one classifier fails, so the others still get trained.
    failed_classifiers = [
        classifier for classifier in classifiers
        if not run_classifier(classifier, environment)
    ]

    if failed_classifiers:
        print(f"[ERROR] Failed classifiers: {', '.join(failed_classifiers)}. Aborting pipeline.")
        sys.exit(1)

    print('[PIPELINE] Step 1 completed successfully.\n')


# ---------------------------------------------------------------------------
# Step 2: distribute the generated models
# ---------------------------------------------------------------------------

def copy_model_files():
    """Step 2: Distribute the generated models and headers to the ESP32 project."""
    print('[PIPELINE] Step 2: Copying model files to ESP32 firmware folder...')

    if not models_directory.exists():
        print(f'[ERROR] Models directory not found: {models_directory}')
        sys.exit(1)
    if not firmware_directory.exists():
        print(f'[ERROR] Firmware directory not found: {firmware_directory}')
        sys.exit(1)

    firmware_src_directory.mkdir(parents=True, exist_ok=True)

    # 2.1 .tflite files go to the firmware root (embedded through CMakeLists.txt)
    for file_name in tflite_files:
        source = models_directory / file_name
        if source.exists():
            shutil.copy(source, firmware_directory / file_name)
            print(f'  -> Copied {file_name} to firmware root ({source.stat().st_size / 1024:.0f} KB).')
        else:
            print(f'[WARNING] {file_name} not found in models directory.')

    # 2.2 C/C++ headers go to the firmware src/ directory
    for file_name in header_files:
        source = models_directory / file_name
        if source.exists():
            shutil.copy(source, firmware_src_directory / file_name)
            print(f'  -> Copied {file_name} to firmware src folder.')
        else:
            print(f'[WARNING] {file_name} not found in models directory.')

    print('[PIPELINE] Step 2 completed successfully.\n')


def clean_model_files():
    """Reverses copy_model_files by deleting the models and headers from the ESP32 project."""
    print('[PIPELINE] Deleting model files from ESP32 firmware folder...')

    if not firmware_directory.exists():
        print(f'[WARNING] Firmware directory not found: {firmware_directory}')
        return

    for file_name in tflite_files:
        target = firmware_directory / file_name
        if target.exists():
            target.unlink()
            print(f'  -> Deleted {file_name} from firmware root.')
        else:
            print(f'  -> [SKIP] {file_name} not found in firmware root.')

    for file_name in header_files:
        target = firmware_src_directory / file_name
        if target.exists():
            target.unlink()
            print(f'  -> Deleted {file_name} from firmware src folder.')
        else:
            print(f'  -> [SKIP] {file_name} not found in firmware src folder.')

    print('[PIPELINE] Deletion completed successfully.\n')


def clean_training_directories():
    """Deletes the logs and models directories and all their contents."""
    print('[PIPELINE] Deleting ML output directories...')
    
    ml_dir = Path(__file__).resolve().parent
    directories_to_delete = ['logs', 'models']
    
    for dir_name in directories_to_delete:
        target_dir = ml_dir / dir_name
        
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            print(f'  -> Deleted directory and its contents: {dir_name}/')
        else:
            print(f'  -> [SKIP] Directory not found: {dir_name}/')

    print('[PIPELINE] Directories deletion completed successfully.\n')


# ---------------------------------------------------------------------------
# Step 3: firmware
# ---------------------------------------------------------------------------

def find_pio_executable():
    """Returns the path to the PlatformIO CLI, or None if it cannot be found."""
    pio = shutil.which('pio') or shutil.which('platformio')
    if pio:
        return pio

    # Fallback: default PlatformIO install location (used by the VS Code extension)
    scripts_dir = 'Scripts' if sys.platform == 'win32' else 'bin'
    for name in ('pio', 'platformio'):
        candidate = Path.home() / '.platformio' / 'penv' / scripts_dir / name
        for path in (candidate, candidate.with_suffix('.exe')):
            if path.exists():
                return str(path)
    return None


def build_and_upload_firmware(model):
    """Step 3: Build and flash the ESP32 firmware for a single model.

    Args:
        model: One of 'cnn', 'mlp', 'rf' or 'svm'. Must match an environment
               name in esp32_firmware/platformio.ini. Only one model fits on
               the board at a time.
    """
    model = model.lower()
    if model not in classifier_scripts:
        print(f"[ERROR] Invalid model '{model}'. Choose one of: {', '.join(classifier_scripts)}.")
        sys.exit(1)

    print(f"[PIPELINE] Step 3: Building and uploading ESP32 firmware for model '{model}'...")

    pio = find_pio_executable()
    if pio is None:
        print("[ERROR] 'pio' command not found. Ensure PlatformIO is installed and in your system PATH.")
        sys.exit(1)

    result = subprocess.run(
        [pio, 'run', '-e', model, '-t', 'upload'],
        cwd=firmware_directory,
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] PlatformIO build/upload failed for model '{model}'.")
        sys.exit(1)

    print(f"[PIPELINE] Step 3 completed successfully for model '{model}'.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print('==================================================')
    print('         TinyML Automated Pipeline Start          ')
    print('==================================================')

    # Comment or uncomment the steps you want to run!

    # Step 1: train the models. Accepts 'cnn', 'mlp', 'rf' and/or 'svm'
    run_trainings(['cnn', 'mlp', 'rf', 'svm'])

    # Step 2: copy the generated models into the firmware project.
    copy_model_files()
    # clean_model_files()
    # clean_training_directories()

    # Step 3: build and flash one model. Only one fits on the board at a time.
    # build_and_upload_firmware('cnn')

    print('==================================================')
    print('        TinyML Automated Pipeline Finished!       ')
    print('==================================================')


if __name__ == '__main__':
    main()