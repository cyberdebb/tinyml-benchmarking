import subprocess
import shutil
import sys
from pathlib import Path


def run_trainings():
    """Step 1: Execute the all_trainings.py script."""
    print("[PIPELINE] Step 1: Running all ML trainings...")
    
    # sys.executable ensures we use the same Python interpreter currently running
    result = subprocess.run([sys.executable, "src/all_trainings.py"])
    
    if result.returncode != 0:
        print("[ERROR] Training failed. Aborting pipeline.")
        sys.exit(1)
        
    print("[PIPELINE] Step 1 completed successfully.\n")


def copy_model_files():
    """Step 2: Distribute the generated models and headers to the ESP32 project."""
    print("[PIPELINE] Step 2: Copying model files to ESP32 firmware folder...")
    
    # Define paths based on the current script's location
    ml_dir = Path(__file__).resolve().parent
    models_dir = ml_dir / "models"
    firmware_dir = ml_dir.parent / "esp32_firmware"
    firmware_src_dir = firmware_dir / "src"

    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        sys.exit(1)
    if not firmware_dir.exists():
        print(f"[ERROR] Firmware directory not found: {firmware_dir}")
        sys.exit(1)

    # 2.1 Copy .tflite files to the firmware root (for platformio.ini embedding)
    tflite_files = ["cnn_classifier.tflite", "mlp_classifier.tflite"]
    for file_name in tflite_files:
        src = models_dir / file_name
        dst = firmware_dir / file_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  -> Copied {file_name} to firmware root.")
        else:
            print(f"[WARNING] {file_name} not found in models directory.")

    # 2.2 Copy C/C++ header files to the firmware src/ directory
    header_files = [
        "svm_classifier.h", 
        "svm_classifier_scaler.h",
        "random_forest_classifier.h",
        "random_forest_model.h"
    ]
    for file_name in header_files:
        src = models_dir / file_name
        dst = firmware_src_dir / file_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  -> Copied {file_name} to firmware src folder.")
        else:
            print(f"[WARNING] {file_name} not found in models directory.")
            
    print("[PIPELINE] Step 2 completed successfully.\n")


def build_and_upload_firmware():
    """Step 3: Call PlatformIO to build and flash the ESP32."""
    print("[PIPELINE] Step 3: Building and uploading ESP32 firmware...")
    
    ml_dir = Path(__file__).resolve().parent
    firmware_dir = ml_dir.parent / "esp32_firmware"

    try:
        # shell=True is highly recommended on Windows to properly resolve the 'pio' command
        result = subprocess.run(
            ["pio", "run", "-t", "upload"], 
            cwd=firmware_dir,
            shell=True 
        )
        if result.returncode != 0:
            print("[ERROR] PlatformIO build/upload failed.")
            sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'pio' command not found. Ensure PlatformIO is installed and in your system PATH.")
        sys.exit(1)
        
    print("[PIPELINE] Step 3 completed successfully.\n")


def delete_model_files():
    """Reverses copy_model_files by deleting the models and headers from the ESP32 project."""
    print("[PIPELINE] Deleting model files from ESP32 firmware folder...")
    
    # Define paths based on the current script's location
    ml_dir = Path(__file__).resolve().parent
    firmware_dir = ml_dir.parent / "esp32_firmware"
    firmware_src_dir = firmware_dir / "src"

    if not firmware_dir.exists():
        print(f"[WARNING] Firmware directory not found: {firmware_dir}")
        return

    # 1. Delete .tflite files from the firmware root
    tflite_files = ["cnn_classifier.tflite", "mlp_classifier.tflite"]
    for file_name in tflite_files:
        target_file = firmware_dir / file_name
        if target_file.exists():
            target_file.unlink() # Deletes the file
            print(f"  -> Deleted {file_name} from firmware root.")
        else:
            print(f"  -> [SKIP] {file_name} not found in firmware root.")

    # 2. Delete C/C++ header files from the firmware src/ directory
    header_files = [
        "svm_classifier.h", 
        "svm_classifier_scaler.h",
        "random_forest_classifier.h",
        "random_forest_model.h"
    ]
    for file_name in header_files:
        target_file = firmware_src_dir / file_name
        if target_file.exists():
            target_file.unlink() # Deletes the file
            print(f"  -> Deleted {file_name} from firmware src folder.")
        else:
            print(f"  -> [SKIP] {file_name} not found in firmware src folder.")
            
    print("[PIPELINE] Deletion completed successfully.\n")
    
    
def main():
    print("==================================================")
    print("         TinyML Automated Pipeline Start          ")
    print("==================================================")
    
    # Uncomment these lines according to your needs!
    
    # run_trainings()
    copy_model_files()
    # delete_model_files()
    # build_and_upload_firmware()
    
    print("==================================================")
    print("        TinyML Automated Pipeline Finished!       ")
    print("==================================================")

if __name__ == "__main__":
    main()