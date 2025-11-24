import os
import sys
import io
import traceback
from fickling.pytorch import PyTorchModelWrapper
from fickling.fickle import Pickled, Interpreter
from fickling.tracing import Trace
import zipfile
import pandas as pd
from download_injected import do_open_source_checks

SUPPORTED_EXTENSIONS = (".bin", ".pkl", ".pickle", ".pt", ".pth", ".data", ".th")


def is_zip_model(path):
    try:
        with zipfile.ZipFile(path, "r") as zip_test:
            return True
    except zipfile.BadZipFile:
        return False


## Walk through the entire directory tree
def scannign_directory(base_dir):
    for path, folders, files in os.walk(base_dir):
        # trace_files = [f for f in files if f.lower().endswith(".trace.txt")]
        # if trace_files:
        #     print(f"Skipping {path}, found a .trace.txt in it")
        #     continue
        for filename in files:
            lower_filename = filename.lower()

            if lower_filename.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(path, filename)
                print(f"Processing: {file_path}")
                try:
                    do_open_source_checks(file_path, "scanning_directory.csv")
                except Exception as e:
                    print(f"Failed to process '{file_path}': {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print()


if __name__ == "__main__":
    scannign_directory("/mnt/The_Second_Drive/Security/ML_Research/benign_test")
