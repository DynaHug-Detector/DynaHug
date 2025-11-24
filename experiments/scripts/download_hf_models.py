import argparse
import csv
import pickle
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
import pandas as pd
import requests
import os
import zipfile
import shutil
import psutil
from utils import save_model_metadata, upload_to_gcs
import time
import random
from fickle_insertion import load_payloads_from_csv, inject_and_test_single_model
from download_injected import do_open_source_checks


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# Constants
TAG = None
SPACE_LIMIT_GB = 40
SPACE_LIMIT_BYTES = SPACE_LIMIT_GB * 1024 * 1024 * 1024
LIMIT_MATCHES = 10000
api = HfApi()


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_safety(
    estimated_usage_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5
):
    """
    Check if deserializing the pickle file is safe given memory constraints.
    """
    current_memory_mb = get_memory_usage()

    max_memory_mb = max_memory_gb * 1024
    safety_margin_mb = safety_margin_gb * 1024  # Buffer to avoid OOM errors

    peak_usage_mb = current_memory_mb + estimated_usage_mb * safety_factor
    safe_limit_mb = max_memory_mb - safety_margin_mb

    is_safe = peak_usage_mb <= safe_limit_mb

    return is_safe


def get_remote_file_size(repo_id, filename, repo_type=None):
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            size = int(response.headers.get("content-length", 0))
            return size
    except Exception as e:
        print(f"Error getting size for {repo_id}/{filename}: {e}")
    return None


def get_directory_size(path):
    """Get the total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def get_download_size(analysis_type):
    download_count = 0
    total_size = 0

    for model in models:
        if download_count >= LIMIT_MATCHES:
            print(f"\nReached max download limit of {LIMIT_MATCHES}. Stopping.")
            break

        model_id = model.modelId
        try:
            print(f"\nChecking model: {model_id}")
            info = None
            if analysis_type == "model":
                info = api.model_info(model_id)
            elif analysis_type == "dataset":
                info = api.dataset_info(model_id)
            else:
                print("Invalid type. Please try again")
                return
            siblings = info.siblings

            # Look for .bin files first (required for the workflow)
            file = None

            for file in siblings:
                fname = file.rfilename.lower()
                if analysis_type == "model" and fname == "pytorch_model.bin":
                    file = fname
                    break
                elif analysis_type == "dataset" and fname.endswith(".py"):
                    file = fname
            if not file:
                print("No pytorch_model.bin file found. Skipping this model.")
                continue

            download_count += 1
            # Get and print the remote file size
            size_bytes = get_remote_file_size(model_id, file)
            if size_bytes is not None:
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_mb
                print(f"Size of {file}: {size_mb:.2f} MB")
            else:
                print(f"Could not determine size for {file}")
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")

    print(
        f"Total size of all {LIMIT_MATCHES} models would occupy on disk: {total_size}MB"
    )


def extract_and_cleanup(bin_file_path, extract_dir):
    """
    Extract .bin file and clean up, keeping only .bin and .pkl/.pickle files
    Returns True if pickle files were found and extracted successfully
    """
    try:
        with zipfile.ZipFile(bin_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
            print(f"Extracted contents to: {extract_dir}")

        # Collect all .pkl/.pickle files
        pkl_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if (
                    file.endswith(".pkl")
                    or file.endswith(".pickle")
                    or file.endswith(".pt")
                ):
                    pkl_files.append(os.path.join(root, file))

        if not pkl_files:
            print("No .pkl or .pickle files found after extraction.")
            return False

        print(f"Found {len(pkl_files)} pickle files. Cleaning up...")

        bin_filename = os.path.basename(bin_file_path)
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for file in files:
                fpath = os.path.join(root, file)
                if not (file == bin_filename):
                    os.remove(fpath)
            for d in dirs:
                dirpath = os.path.join(root, d)
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)

        return True

    except zipfile.BadZipFile:
        print("Not a zip archive.")
        return False
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def download_models():
    download_count = 0

    for model in models:
        if download_count >= LIMIT_MATCHES:
            print(f"\nReached max download limit of {LIMIT_MATCHES}. Stopping.")
            break

        model_id = model.modelId
        try:
            print(f"\nChecking model: {model_id}")
            info = api.model_info(model_id)
            siblings = info.siblings

            # Look for .bin files first (required for the workflow)
            pytorch_bin_file = None

            for file in siblings:
                fname = file.rfilename.lower()
                if fname == "pytorch_model.bin":
                    pytorch_bin_file = fname
                    break

            if not pytorch_bin_file:
                print("No pytorch_model.bin file found. Skipping this model.")
                continue

            print(f"Found .bin file: {pytorch_bin_file}. Downloading...")
            local_dir = os.path.join(DOWNLOAD_DIR, model_id.replace("/", "__"))

            # Download the .bin file
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[pytorch_bin_file],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

            bin_path = os.path.join(local_dir, pytorch_bin_file)

            # Try to extract and find pickle files
            success = extract_and_cleanup(bin_path, local_dir)

            if success:
                print(f"Successfully processed {model_id}")
                print(f"   - Kept .bin file: {bin_path}")
                print(f"   - Extracted pickle files")
                print(f"   - Cleaned up other files")
                download_count += 1
            else:
                print(
                    f"Could not extract pickle files from {model_id}. Removing the folder"
                )
                shutil.rmtree(local_dir)

        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")

    print(f"\nSuccessfully downloaded and processed {download_count} models.")


# Helper function for payload cycling
def get_cycled_payload(final_list, payload_mapping, total_injections):
    """Get the next payload in cycle, with detailed information"""
    if not final_list:
        return None, None, {}

    current_payload_index = total_injections % len(final_list)
    payload = final_list[current_payload_index]
    cycle_number = total_injections // len(final_list) + 1
    position_in_cycle = (total_injections % len(final_list)) + 1

    payload_info = {
        "payload_index": current_payload_index,
        "cycle_number": cycle_number,
        "position_in_cycle": position_in_cycle,
        "total_injection_number": total_injections,
    }

    return payload, current_payload_index, payload_info


def download_and_upload_injected(
    download_dir,
    space_limit_bytes,
    TAG,
    explored_models,
    download_log_df,
    limit,
    direction="desc",
    gcs_bucket_name=None,  # New parameter for GCS upload
    models_to_download=[],
):
    """Modified download function that injects payloads immediately after each download and uploads to GCS."""

    final_list, payload_mapping = load_payloads_from_csv("../malhug_result_info.csv")
    payload_index = 0
    total_injections = 0

    print(f"Loaded {len(final_list)} payloads")
    if models_to_download == []:
        if TAG is None:
            models = list(api.list_models(sort="likes"))
        else:
            models = list(api.list_models(filter=TAG, sort="likes"))
    else:
        models = models_to_download
    if direction == "asc":
        models = list(reversed(models))

    model_ids = [model.id for model in models]

    # Find the index of the target model
    # uncomment if you want to download froma  particular model index
    # target_model = "csebuetnlp/banglat5_nmt_en_bn"
    #
    # try:
    #     index = model_ids.index(target_model)
    #     print(f"Index of '{target_model}': {index}")
    # except ValueError:
    #     print(f"Model '{target_model}' not found.")
    #     models.index("csebuetnlp/banglat5_nmt_en_bn")

    index = 0
    downloaded_models = []
    injection_results = []
    current_size = 0
    i = 0
    fail_count = 0
    start = index

    if not download_log_df.empty:
        last_download = download_log_df.iloc[-1]["name"]
        for model in models:
            i += 1
            if model.modelId == last_download:
                start = i + 1
                break

    for j in range(start, len(models)):
        model_id = models[j].modelId
        if "TransQuest" in model_id:
            continue
        if total_injections >= 400:
            break
        if len(explored_models) + len(downloaded_models) >= limit:
            print(f"{limit} models have been downloaded")
            break

        if model_id in explored_models and direction == "asc":
            print("Encountered an already explored model in the wild run. Stopping.")
            break

        i += 1
        if model_id in explored_models:
            print(f"Skipping already explored model: {model_id}")
            continue

        try:
            print(f"\nChecking model: {model_id}")
            time.sleep(random.uniform(0.2, 0.75))
            info = api.model_info(model_id)
            siblings = info.siblings

            pytorch_bin_file = next(
                (
                    file.rfilename
                    for file in siblings
                    if file.rfilename.lower() == "pytorch_model.bin"
                ),
                None,
            )

            if not pytorch_bin_file:
                print("No pytorch_model.bin file found. Skipping this model.")
                continue

            file_size = get_remote_file_size(model_id, pytorch_bin_file)
            if file_size is None:
                print(
                    f"Skipping model. Could not determine size for {pytorch_bin_file}"
                )
                continue

            if file_size * 2 > space_limit_bytes:
                print(
                    f"Adding {model_id} would exceed space limit. Stopping downloads."
                )
                break

            file_size_mb = file_size / (1024 * 1024)
            is_safe = check_memory_safety(
                file_size_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5
            )

            if not is_safe:
                print(
                    f"Memory safety check failed for {model_id} with size {file_size_mb}. Skipping download."
                )
                continue

            print(f"Found .bin file: {pytorch_bin_file}. Size: {file_size_mb:.2f} MB")
            local_dir = os.path.join(download_dir, model_id.replace("/", "__"))

            pytorch_bfile = os.path.join(local_dir, "pytorch_model.bin")
            if os.path.exists(pytorch_bfile):
                print(f"Model {model_id} found in cache. Skipping download.")
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                downloaded_models.append(model_id)
                continue

            snapshot_download(
                repo_id=model_id,
                allow_patterns=[pytorch_bin_file],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

            bin_path = os.path.join(local_dir, pytorch_bin_file)
            try:
                do_open_source_checks(
                    bin_path, "no_cluster_benign_set_opensource_results.csv"
                )
            except zipfile.BadZipFile:
                print("Not a zip archive.")
            except Exception as e:
                print(f"Extraction error: {e}")

            success = extract_and_cleanup(bin_path, local_dir)

            if success:
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                download_log_df.loc[len(download_log_df)] = {
                    "name": model_id,
                    "likes": info.likes,
                    "downloads": info.downloads,
                }
                downloaded_models.append(model_id)

                download_log_file_name = (
                    f"wild_{TAG}_downloaded_models.csv"
                    if direction == "asc"
                    else f"{TAG}_downloaded_models.csv"
                )
                save_model_metadata(
                    [download_log_df],
                    [
                        os.path.join(
                            os.path.dirname(download_dir), download_log_file_name
                        )
                    ],
                )

                print(f"Successfully processed {model_id}")
                print(f"   - Actual size: {actual_size / (1024 * 1024):.2f} MB")
                print(
                    f"   - Total size so far: {current_size / (1024 * 1024 * 1024):.2f} GB"
                )
                print(f"Saved CSV file containing model metadata")

                #
            else:
                print(
                    f"Could not extract pickle files from {model_id}. Removing the folder"
                )
                shutil.rmtree(local_dir)
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")
            continue

        try:
            model_path = os.path.join(local_dir, "pytorch_model.bin")

            if os.path.exists(model_path):
                payload, current_payload_index, payload_info = get_cycled_payload(
                    final_list, payload_mapping, total_injections
                )

                if current_payload_index == 0 and total_injections > 0:
                    print(f"Starting new payload cycle #{payload_info['cycle_number']}")

                print(f"Injecting payload into {model_id}...")
                injection_count = 0

                while True:
                    print(
                        f"Using payload [{current_payload_index}] (Cycle {payload_info['cycle_number']}, Position {payload_info['position_in_cycle']}/{len(final_list)})"
                    )
                    total_injections += injection_count
                    payload, current_payload_index, payload_info = get_cycled_payload(
                        final_list, payload_mapping, total_injections
                    )

                    injection_success, load_success, injected_model_path = (
                        inject_and_test_single_model(
                            model_path,
                            current_payload_index,
                            payload,
                            payload_mapping,
                            "no_cluster_injection.csv",
                        )
                    )

                    total_injections -= injection_count
                    if not injection_success or not load_success:
                        injection_count += 1
                        fail_count += 1
                        if fail_count > 5:
                            break
                        continue
                    else:
                        break

                injection_results.append(
                    {
                        "model_id": model_id,
                        **payload_info,
                        "injection_success": injection_success,
                        "load_success": load_success,
                    }
                )

                total_injections += 1
                print(f"Total injections so far: {total_injections}")

                if injection_success and gcs_bucket_name:
                    base_blob_path = (
                        f"injected_models/no_cluster/{model_id.replace('/', '__')}"
                    )
                    original_blob_name = f"{base_blob_path}/pytorch_model.bin"
                    injected_blob_name = f"{base_blob_path}/pytorch_model_injected.bin"

                    upload_to_gcs(model_path, gcs_bucket_name, original_blob_name)

                    upload_to_gcs(
                        injected_model_path, gcs_bucket_name, injected_blob_name
                    )

                downloaded_models.append(model_id)
                shutil.rmtree(local_dir)
                print(total_injections)
        except Exception as e:
            print(f"Error processing {model_id}: {e}")

    total_cycles_completed = total_injections // len(final_list)
    remaining_in_current_cycle = total_injections % len(final_list)

    print(f"\nInjection Summary:")
    print(f"   Total models downloaded: {len(downloaded_models)}")
    print(f"   Total injections performed: {total_injections}")
    print(f"   Complete payload cycles: {total_cycles_completed}")
    print(f"   Payloads used in current cycle: {remaining_in_current_cycle}")

    return downloaded_models, injection_results


def download_models_with_immediate_injection(
    download_dir,
    space_limit_bytes,
    TAG,
    explored_models,
    download_log_df,
    limit,
    direction="desc",
):
    """Modified download function that injects payloads immediately after each download"""

    # Load payloads once at the beginning
    final_list, payload_mapping = load_payloads_from_csv("../malhug_result_info.csv")
    payload_index = 0
    total_injections = 0  # Track total number of injections for cycling

    print(f"Loaded {len(final_list)} payloads")

    # Your existing model discovery logic...
    models = list(api.list_models(filter=TAG, sort="likes"))
    if direction == "asc":
        models = list(reversed(models))

    downloaded_models = []
    injection_results = []
    current_size = 0

    downloaded_models = []
    current_size = 0
    i = 0
    start = 0

    # Resuming from last successful download
    if not download_log_df.empty:
        last_download = download_log_df.iloc[-1]["name"]
        for model in models:
            i += 1
            if model.modelId == last_download:
                start = i + 1
                break

    for j in range(start, len(models)):
        model_id = models[j].modelId
        if len(explored_models) + len(downloaded_models) >= limit:
            print(f"{limit} models have been downloaded")
            break

        # If it is a wild run, we are going in the opposite direction
        # If an already explored model is encountered, we break the loop
        # since it means we are moving into the benign set the model is trained on
        if model_id in explored_models and direction == "asc":
            print("Encountered an already explored model in the wild run. Stopping.")
            break

        i += 1
        if model_id in explored_models:
            print(f"Skipping already explored model: {model_id}")
            actual_size = get_directory_size(local_dir)
            current_size += actual_size
            continue

        try:
            print(f"\nChecking model: {model_id}")
            time.sleep(random.uniform(0.2, 0.75))  # Random delay to avoid rate limiting
            info = api.model_info(model_id)
            siblings = info.siblings

            pytorch_bin_file = None
            for file in siblings:
                fname = file.rfilename.lower()
                if fname == "pytorch_model.bin":
                    pytorch_bin_file = fname
                    break

            if not pytorch_bin_file:
                print("No pytorch_model.bin file found. Skipping this model.")
                continue

            file_size = get_remote_file_size(model_id, pytorch_bin_file)
            if file_size is None:
                print(
                    f"Skipping model. Could not determine size for {pytorch_bin_file}"
                )
                continue

            if current_size + file_size > space_limit_bytes:
                print(
                    f"Adding {model_id} would exceed space limit. Stopping downloads."
                )
                break

            file_size_mb = file_size / (1024 * 1024)
            is_safe = check_memory_safety(
                file_size_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5
            )  # Checking for possible OOM errors

            if not is_safe:
                print(
                    f"Memory safety check failed for {model_id} with size {file_size_mb}. Skipping download."
                )
                continue

            print(f"Found .bin file: {pytorch_bin_file}. Size: {file_size_mb:.2f} MB")
            local_dir = os.path.join(download_dir, model_id.replace("/", "__"))

            pytorch_bfile = os.path.join(local_dir, "pytorch_model.bin")
            if os.path.exists(pytorch_bfile):
                print(f"Model {model_id} found in cache. Skipping download.")
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                downloaded_models.append(model_id)
                continue

            snapshot_download(
                repo_id=model_id,
                allow_patterns=[pytorch_bin_file],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

            bin_path = os.path.join(local_dir, pytorch_bin_file)

            # Try to extract and find pickle files
            success = extract_and_cleanup(bin_path, local_dir)

            if success:
                # Calculate actual size after extraction
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                download_log_df.loc[len(download_log_df)] = {
                    "name": model_id,
                    "likes": info.likes,
                    "downloads": info.downloads,
                }
                downloaded_models.append(model_id)
                download_log_file_name = (
                    f"wild_{TAG}_downloaded_models.csv"
                    if direction == "asc"
                    else f"{TAG}_downloaded_models.csv"
                )
                save_model_metadata(
                    [download_log_df],
                    [
                        os.path.join(
                            os.path.dirname(download_dir), download_log_file_name
                        )
                    ],
                )
                print(f"Successfully processed {model_id}")
                print(f"   - Actual size: {actual_size / (1024 * 1024):.2f} MB")
                print(
                    f"   - Total size so far: {current_size / (1024 * 1024 * 1024):.2f} GB"
                )
                print(f"Saved CSV file containing model metadata")

            else:
                print(
                    f"Could not extract pickle files from {model_id}. Removing the folder"
                )
                shutil.rmtree(local_dir)
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")
            continue

        try:
            # After successful download, immediately inject
            model_path = os.path.join(local_dir, "pytorch_model.bin")

            if os.path.exists(model_path):
                # Get cycled payload
                payload, current_payload_index, payload_info = get_cycled_payload(
                    final_list, payload_mapping, total_injections
                )

                # Check if we're starting a new cycle (back to payload 0 and not the first injection)
                if current_payload_index == 0 and total_injections > 0:
                    print(f"Starting new payload cycle #{payload_info['cycle_number']}")

                print(f"Injecting payload into {model_id}...")
                print(
                    f"Using payload [{current_payload_index}] (Cycle {payload_info['cycle_number']}, Position {payload_info['position_in_cycle']}/{len(final_list)})"
                )

                injection_success, load_success = inject_and_test_single_model(
                    model_path, current_payload_index, payload, payload_mapping
                )

                injection_results.append(
                    {
                        "model_id": model_id,
                        **payload_info,  # Include all payload cycling info
                        "injection_success": injection_success,
                        "load_success": load_success,
                    }
                )

                total_injections += 1
                print(f"Total injections so far: {total_injections}")

            downloaded_models.append(model_id)
            shutil.rmtree(local_dir)

        except Exception as e:
            print(f"Error processing {model_id}: {e}")

    # Summary statistics
    total_cycles_completed = total_injections // len(final_list)
    remaining_in_current_cycle = total_injections % len(final_list)

    print(f"\nInjection Summary:")
    print(f"   Total models downloaded: {len(downloaded_models)}")
    print(f"   Total injections performed: {total_injections}")
    print(f"   Complete payload cycles: {total_cycles_completed}")
    print(f"   Payloads used in current cycle: {remaining_in_current_cycle}")

    return downloaded_models, injection_results


def download_models_with_space_limit(
    download_dir,
    space_limit_bytes,
    TAG,
    explored_models,
    download_log_df,
    limit,
    direction="desc",
):
    """
    Download models within the specified space limit
    """
    models = None
    models = list(api.list_models(filter=TAG, sort="likes"))
    if direction == "asc":
        models = list(reversed(models))
    print(f"Found {len(models)} models with tag '{TAG}'")

    downloaded_models = []
    current_size = 0
    i = 0
    start = 12533
    print(models)
    # exit()

    # Resuming from last successful download
    if not download_log_df.empty:
        last_download = download_log_df.iloc[-1]["name"]
        for model in models:
            i += 1
            if model.modelId == last_download:
                start = i + 1
                break

    for j in range(start, len(models)):
        model_id = models[j].modelId
        if len(explored_models) + len(downloaded_models) >= limit:
            print(f"{limit} models have been downloaded")
            break

        # If it is a wild run, we are going in the opposite direction
        # If an already explored model is encountered, we break the loop
        # since it means we are moving into the benign set the model is trained on
        if model_id in explored_models and direction == "asc":
            print("Encountered an already explored model in the wild run. Stopping.")
            break

        i += 1
        if model_id in explored_models:
            print(f"Skipping already explored model: {model_id}")
            actual_size = get_directory_size(local_dir)
            current_size += actual_size
            continue

        try:
            print(f"\nChecking model: {model_id}")
            time.sleep(random.uniform(0.2, 0.75))  # Random delay to avoid rate limiting
            info = api.model_info(model_id)
            siblings = info.siblings

            pytorch_bin_file = None
            for file in siblings:
                fname = file.rfilename.lower()
                if fname == "pytorch_model.bin":
                    pytorch_bin_file = fname
                    break

            if not pytorch_bin_file:
                print("No pytorch_model.bin file found. Skipping this model.")
                continue

            file_size = get_remote_file_size(model_id, pytorch_bin_file)
            if file_size is None:
                print(
                    f"Skipping model. Could not determine size for {pytorch_bin_file}"
                )
                continue

            if current_size + file_size > space_limit_bytes:
                print(
                    f"Adding {model_id} would exceed space limit. Stopping downloads."
                )
                continue

            file_size_mb = file_size / (1024 * 1024)
            is_safe = check_memory_safety(
                file_size_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5
            )  # Checking for possible OOM errors

            if not is_safe:
                print(
                    f"Memory safety check failed for {model_id} with size {file_size_mb}. Skipping download."
                )
                continue

            print(f"Found .bin file: {pytorch_bin_file}. Size: {file_size_mb:.2f} MB")
            local_dir = os.path.join(download_dir, model_id.replace("/", "__"))
            pytorch_bfile = os.path.join(local_dir, "pytorch_model.bin")
            if os.path.exists(pytorch_bfile):
                print(f"Model {model_id} found in cache. Skipping download.")
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                downloaded_models.append(model_id)
                continue

            snapshot_download(
                repo_id=model_id,
                allow_patterns=[pytorch_bin_file],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )

            bin_path = os.path.join(local_dir, pytorch_bin_file)

            # Try to extract and find pickle files
            success = extract_and_cleanup(bin_path, local_dir)

            if success:
                # Calculate actual size after extraction
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                download_log_df.loc[len(download_log_df)] = {
                    "name": model_id,
                    "likes": info.likes,
                    "downloads": info.downloads,
                }
                downloaded_models.append(model_id)
                download_log_file_name = (
                    f"wild_{TAG}_downloaded_models.csv"
                    if direction == "asc"
                    else f"{TAG}_downloaded_models.csv"
                )
                save_model_metadata(
                    [download_log_df],
                    [
                        os.path.join(
                            os.path.dirname(download_dir), download_log_file_name
                        )
                    ],
                )
                print(f"Successfully processed {model_id}")
                print(f"   - Actual size: {actual_size / (1024 * 1024):.2f} MB")
                print(
                    f"   - Total size so far: {current_size / (1024 * 1024 * 1024):.2f} GB"
                )
                print(f"Saved CSV file containing model metadata")

            else:
                print(
                    f"Could not extract pickle files from {model_id}. Removing the folder"
                )
                shutil.rmtree(local_dir)
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")

    print(f"\nSuccessfully downloaded and processed {len(downloaded_models)} models.")
    print(f"Total space used: {current_size / (1024 * 1024 * 1024):.2f} GB")

    return downloaded_models, i


def download_datasets():
    datasets = list(api.list_datasets(filter=TAG, limit=500, sort="likes"))
    print(f"Found {len(datasets)} datasets with tag '{TAG}'")
    filtered_datasets = []
    for dataset in datasets:
        dataset_id = dataset.id
        try:
            print(f"\nChecking dataset: {dataset_id}")
            info = api.dataset_info(dataset_id)
            siblings = info.siblings

            for file in siblings:
                fname = file.rfilename.lower()
                if fname.endswith(".py"):
                    size_bytes = get_remote_file_size(
                        dataset_id, fname, repo_type="dataset"
                    )
                    if size_bytes is None:
                        continue
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"Found file: {fname} - Size: {size_mb:.2f} MB")
                    if size_mb <= SPACE_LIMIT_GB * 1024:
                        filtered_datasets.append((dataset_id, fname, size_mb))
                        print(f"✔ Added: {dataset_id}/{fname} ({size_mb:.2f} MB)")
                        break

            if len(filtered_datasets) >= LIMIT_MATCHES:
                break

        except Exception as e:
            print(f"⚠️ Skipping {dataset_id} due to error: {e}")
            continue

    print(f"\n{len(filtered_datasets)} datasets passed the filter.")

    for dataset_id, filename, size_mb in filtered_datasets:
        print(f"Downloading {dataset_id}/{filename} ({size_mb:.2f} MB)...")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            allow_patterns=[filename],
            local_dir=os.path.join(
                DOWNLOAD_DIR, "dataset", dataset_id.replace("/", "__")
            ),
            local_dir_use_symlinks=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download models or datasets from Hugging Face Hub."
    )
    parser.add_argument(
        "--type",
        choices=["model", "dataset"],
        required=True,
        help="Specify what to download: 'model' or 'dataset'",
    )
    parser.add_argument(
        "--base-dir", required=True, help="The place to download the files into"
    )
    parser.add_argument(
        "--tag",
        default=TAG,
        help="Tag to filter models or datasets (default: 'text-generation-inference')",
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="Tag to find how much space LIMIT_MATCHES models or datasets would take if downloaded",
    )
    parser.add_argument(
        "--upload", action="store_true", help="to upload, or not to uplaod"
    )
    parser.add_argument(
        "--list-path",
        help="point to pickle with model names to download/check",
    )
    parser.add_argument("--bucket_name", help="bucket name for gcs")
    args = parser.parse_args()
    models = []

    if args.tag:
        TAG = args.tag
    if args.list_path:
        models = []
        with open(args.list_path, "rb") as file:
            models = pickle.load(file)

        print("Loaded models:", models[:10])
    DOWNLOAD_DIR = args.base_dir
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if args.size:
        get_download_size(args.type)
    elif args.type == "model":
        download_log_path = os.path.join(DOWNLOAD_DIR, f"{TAG}_downloaded_models.csv")

        if os.path.exists(download_log_path):
            download_log_df = pd.read_csv(download_log_path)
            explored_models = set(download_log_df["name"].tolist())
        else:
            download_log_df = pd.DataFrame(columns=["name", "likes", "downloads"])
            explored_models = set()
        if args.upload:
            print("reaching here")
            if args.bucket_name:
                BUCKET_NAME = args.bucket_name
                downloaded_models, injected_models = download_and_upload_injected(
                    download_dir=DOWNLOAD_DIR,
                    space_limit_bytes=SPACE_LIMIT_BYTES,
                    TAG=TAG,
                    explored_models=explored_models,
                    download_log_df=download_log_df,
                    limit=LIMIT_MATCHES,
                    direction="desc",
                    gcs_bucket_name=BUCKET_NAME,
                    models_to_download=models,
                )
            else:
                print("Please provide bucket name for gcp uploading")

        else:
            downloaded_models, injected_models = (
                download_models_with_immediate_injection(
                    download_dir=DOWNLOAD_DIR,
                    space_limit_bytes=SPACE_LIMIT_BYTES,
                    TAG=TAG,
                    explored_models=explored_models,
                    download_log_df=download_log_df,
                    limit=LIMIT_MATCHES,
                    direction="desc",  # or "asc" for wild run
                )
            )
        print(f"\nDownloaded {len(downloaded_models)} models.")
        download_log_df.to_csv(download_log_path, index=False)

    elif args.type == "dataset":
        download_datasets()
    else:
        print("Invalid type specified. Use 'model' or 'dataset'.")
