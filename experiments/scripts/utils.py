from pathlib import Path
import tarfile
from google.cloud import storage
import os
import pandas as pd
import traceback
import re
import psutil
import platform
import subprocess
import resource
from huggingface_hub import HfApi

"""
Parsing the filenames from the dataframe(malhug_result_info.csv) 
"""


def parse_files(files):
    res = []
    file_list = files.split(",")
    for file in file_list:
        idx = file.find(":")
        res.append(
            file[: idx if idx != -1 else len(file)]
        )  # Get the filename before the colon if it exists
    return res


def parse_feature_from_file(file_path):
    """
    Parses the list of all syscalls/opcodes from a file.
    """
    syscalls = []
    with open(file_path, "r") as f:
        for line in f:
            syscall = line.strip()
            if syscall:  # Ensure the line is not empty
                syscalls.append(syscall)
    return syscalls


def parse_syscalls_from_file(file_path):
    """
    Parses the list of all syscalls/opcodes from a file.
    """
    syscalls = []
    with open(file_path, "r") as f:
        for line in f:
            syscall = line.strip()
            if syscall:  # Ensure the line is not empty
                syscalls.append(syscall)
    return syscalls


def save_model_metadata(dfs, file_paths):
    try:
        for df, path in zip(dfs, file_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Saved metadata to {path}")
    except Exception:
        print("An error occurred while saving the csv")
        traceback.print_exc()


def upload_to_gcs(file_path, bucket_name, blob_name):
    try:
        client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        print(f"Uploading {file_path} to GCP bucket {bucket_name} as {blob.name}")
        blob.upload_from_filename(file_path)

        print(f"Successfully uploaded to GCP bucket {bucket_name}")
        return True

    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False


def list_files_gcs(bucket_name, prefix=None):
    """
    List of all files in a GCS bucket
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    res = [blob.name for blob in blobs]
    return res


def download_from_gcs(bucket_name, blob_name, destination_file_name):
    """
    Download a file from Google Cloud Storage
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        print(
            f"Downloading {blob_name} from GCP bucket {bucket_name} to {destination_file_name}"
        )
        blob.download_to_filename(destination_file_name)

        print(f"Successfully downloaded to {destination_file_name}")
        return True

    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return False
