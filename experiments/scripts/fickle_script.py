import os
import argparse
import sys
import io
import traceback
from fickling.pytorch import PyTorchModelWrapper
from fickling.fickle import Pickled, Interpreter
from fickling.tracing import Trace
import zipfile
import pandas as pd


# Supported file extensions
SUPPORTED_EXTENSIONS = (".bin", ".pkl", ".pickle", ".pt", ".pth", ".data", ".th")


def is_zip_model(path):
    try:
        with zipfile.ZipFile(path, "r") as zip_test:
            return True
    except zipfile.BadZipFile:
        return False


def without_filtering(base_dir):
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
                    if is_zip_model(file_path):
                        fickled_model = PyTorchModelWrapper(file_path, force=True)
                        pickles = fickled_model.pickled
                    else:
                        with open(file_path, "rb") as f:
                            pickles = Pickled.load(f)

                    # Setup interpreter and tracer
                    interpreter = Interpreter(pickles)
                    trace = Trace(interpreter)

                    # Capture printed trace.run() output
                    buffer = io.StringIO()
                    sys_stdout = sys.stdout
                    sys.stdout = buffer
                    try:
                        trace.run()
                    finally:
                        sys.stdout = sys_stdout

                    trace_output = buffer.getvalue()

                    trace_file = file_path + ".trace.txt"
                    with open(trace_file, "w") as f:
                        f.write(trace_output)

                    print(f"Saved trace to: {trace_file}")

                except Exception as e:
                    print(f"Failed to process '{file_path}': {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print()


def with_filtering(base_dir, csv_path):
    if csv_path:
        ben_set = pd.read_csv(
            csv_path,
        )
        ben_model_names = ben_set["name"]
        print(ben_model_names)
    else:
        print("Please provide the csv path to filter with")
        exit()
    for path, folders, files in os.walk(base_dir):
        # trace_files = [f for f in files if f.lower().endswith(".trace.txt")]
        # if trace_files:
        #     print(f"Skipping {path}, found a .trace.txt in it")
        #     continue
        for filename in files:
            lower_filename = filename.lower()

            if lower_filename.endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(path, filename)
                for model_name in ben_model_names:
                    # print(model_name)
                    # print(file_path)
                    if model_name.replace("/", "__") in file_path:  # <-- suffix match
                        print(f"Processing: {file_path}")
                        pass
                        try:
                            if is_zip_model(file_path):
                                fickled_model = PyTorchModelWrapper(
                                    file_path, force=True
                                )
                                pickles = fickled_model.pickled
                            else:
                                with open(file_path, "rb") as f:
                                    pickles = Pickled.load(f)

                            # Setup interpreter and tracer
                            interpreter = Interpreter(pickles)
                            trace = Trace(interpreter)

                            # Capture printed trace.run() output
                            buffer = io.StringIO()
                            sys_stdout = sys.stdout
                            sys.stdout = buffer
                            try:
                                trace.run()
                            finally:
                                sys.stdout = sys_stdout

                            trace_output = buffer.getvalue()

                            trace_file = file_path + ".trace.txt"
                            with open(trace_file, "w") as f:
                                f.write(trace_output)

                            print(f"Saved trace to: {trace_file}")

                        except Exception as e:
                            print(
                                f"Failed to process '{file_path}': {type(e).__name__}: {e}"
                            )
                            traceback.print_exc()
                            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run fickling on a directory, with filtering based on a csv or not"
    )
    parser.add_argument("--base-dir", required=True, help="directory to run traces of")
    parser.add_argument(
        "--filtered", action="store_true", help="filtered by csv, use flag if yes"
    )
    parser.add_argument(
        "--csv-path",
        help="path to csv to filter with, has to have a name column that can be used for filtering",
    )
    args = parser.parse_args()
    if args.filtered:
        with_filtering(args.base_dir, args.csv_path)
        exit()
    else:
        without_filtering(args.base_dir)
