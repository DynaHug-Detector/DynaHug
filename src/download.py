import argparse
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
import requests
import os
import zipfile
import shutil
import psutil
from utils.utils import save_model_metadata, get_model_metadata
import time
import random
import json
import traceback
import sys
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
import pickle
import sys
import re

# Constants
TAG = None
HF_ENDPOINT = "https://hf-mirror.com"
SPACE_LIMIT_GB = 30  # Space limit in GB
SPACE_LIMIT_BYTES = SPACE_LIMIT_GB * 1024 * 1024 * 1024  # Convert to bytes
LIMIT_MATCHES = 50
API_DETECTION_LOG_DIR = "malicious_detection_logs"
SUPPORTED_FORMATS = {".pickle", ".pkl", ".pt", ".pth", ".bin", ".th", ".data", ".joblib", ".dill"} # All extensions which could be loaded by pickle.

os.makedirs(API_DETECTION_LOG_DIR, exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" # To enable faster downloads

api = HfApi()

# The problem is that we can't use the same if isinstance(state_dict, nn.Module) wouldnt work for all pickle files
# Which is why we need to identify which are sharded pickles and avoid being passed into this function
# Robust way is to use the .index.<something maybe blank>.json
# Extract the names of the shards from this file and then exclude from the check
# Don't need to check if the shards are pickles since they inherently are made from torch serialization

class ModelCache:
    def __init__(self):
        self._cache = {}

    def getModelList(self, tag, sort, direction):
        """Get models list from cache or fetch from API"""
        cache_key = f"{tag}_{sort}_{direction}"
        if cache_key not in self._cache:
            print(f"Fetching models with tag '{tag}' from Hugging Face Hub...")
            models = list(api.list_models(filter=tag if tag != 'all' else None, sort=sort))
            if direction == "asc":
                models = list(reversed(models))
            self._cache[cache_key] = models
            print(f"Found {len(models)} models with tag '{tag}'")
        return self._cache[cache_key]

model_cache = ModelCache()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_safety(estimated_usage_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5):
    """
    Check if deserializing the pickle file is safe given memory constraints.
    """
    current_memory_mb = get_memory_usage()
    
    max_memory_mb = max_memory_gb * 1024
    safety_margin_mb = safety_margin_gb * 1024 # Buffer to avoid OOM errors
    
    peak_usage_mb = current_memory_mb + estimated_usage_mb * safety_factor
    safe_limit_mb = max_memory_mb - safety_margin_mb
    
    is_safe = peak_usage_mb <= safe_limit_mb
    
    return is_safe

def get_remote_file_size(repo_id, filename, repo_type=None):
    try:
        time.sleep(random.uniform(0.7, 1))  # Random delay to avoid rate limiting
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            size = int(response.headers.get("content-length", 0))
            return size
        else:
            print(f"Failed to get size for {repo_id}/{filename}. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error getting size for {repo_id}/{filename}: {e}")
    return None

def get_security_score(security_status):
    """
    Convert security status from hugging face tools to a numerical score.
    For now each unsafe detection is count as 1 point.
    """
    tools = set(["protectAiScan", "avScan", "pickleImportScan", "jFrogScan", "virusTotalScan"])  
    count = {tool : 0 for tool in tools}
    for tool in tools:
        if tool in security_status:
            status = security_status[tool].get("status", "")
            if status == "unsafe":
                count[tool] += 1
    return count

def get_security_info(files, repo_id, tag):
    """
    Get security information of hugging face tools on the pytorch_model.bin file
    """
    ENDPOINT = f"{HF_ENDPOINT}/api/models/{repo_id}/tree/main?expand=True"
    res = {}
    try:
        response = requests.get(ENDPOINT)
        data = response.json()
        isMal = False
        if response.status_code == 200:
            for target_file in files:
                for file_info in data:
                    if target_file in file_info.get("path", ""):
                        security_info = file_info.get("securityFileStatus", {})
                        if security_info.get("status", "") == "unsafe":
                            isMal = True
                            res[target_file] = {
                                "status": "unsafe",
                                "score": get_security_score(security_info),
                            }
                        else:
                            res[target_file] = {
                                "status": security_info.get("status", "unknown"),
                                "score": {},
                            }
            if isMal:
                # Save the file_info for this model into log json file
                log_file_path = os.path.join(API_DETECTION_LOG_DIR, tag, f"{repo_id.replace('/', '__')}_security_info.json")
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, "w") as log_file:
                    json.dump(file_info, log_file, indent=4)

                print(f"Security info for {repo_id} saved to {log_file_path}")
    except Exception as e:
        print(f"Error fetching security info for {repo_id}: {e}")
    return res

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
    
    print(f"Total size of all {LIMIT_MATCHES} models would occupy on disk: {total_size}MB")

def get_checkpoint_names(repo_name, local_dir):
    """
    Handling sharded pickle files present in HF repository
    """
    checkpoint_names = set()

    print("Obtaining the sharded pickle file names")
    shard_index_dir = snapshot_download(
        repo_id=repo_name,
        allow_patterns=[f"*{ext}.index*.json" for ext in SUPPORTED_FORMATS],
        local_dir=local_dir,
    )

    index_files = Path(shard_index_dir).rglob("*.index*.json")
    for shard_file_path in index_files:
        try:
            with open(shard_file_path, 'r') as f:
                data = json.load(f)
                if "weight_map" not in data:
                    print("File doesn't contain weight map. File might not be a pytorch model(pickle).")
                    continue
                checkpoint_names.update(set([
                    str((shard_file_path.parent / fname).relative_to(shard_index_dir))
                    for fname in set(data["weight_map"].values())
                ]))
                print("Shards found:")
                for shard in checkpoint_names:
                    print(f"    - {shard}")
        except Exception as e:
            print(f"An error occurred when decoding the index json file : {e}")
            traceback.print_exc()
    return checkpoint_names

def extract_and_cleanup(bin_file_path, extract_dir):
    """
    Extract .bin file and clean up, keeping only .bin and .pkl/.pickle files.
    Returns True if pickle files were found and extracted successfully.
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

        # Keep only the .bin and pickle files, remove everything else
        bin_filename = os.path.basename(bin_file_path)
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for file in files:
                fpath = os.path.join(root, file)
                # Keep .bin and pickle files, remove everything else
                if not (file == bin_filename):
                # if not (file.endswith(".pkl") or file.endswith(".pickle") or 
                #        file.endswith(".pt") or file == bin_filename):
                    os.remove(fpath)
            # Remove empty directories
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

def isPickle(file_path):
    """
    Checking if .bin and .pt variants have pickle files inside them without extraction.
    Returns True if pickle files were found and extracted successfully.
    """
    try:
        pkl_files = []
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for fname in zip_ref.namelist():
                if fname.endswith((".pkl", ".pickle")):
                    pkl_files.append(fname)
            if not pkl_files:
                print("No .pkl or .pickle files found after extraction.")
                return False

        print(f"Found {len(pkl_files)} pickle files.")
        for fname in pkl_files:
            print(f" - {fname}")
        return True

    except zipfile.BadZipFile:
        print("Not a zip archive.")
        return False
    except Exception as e:
        print(f"Extraction error: {e}")
        return False

def isPickle_improved(file_path):
    try:
        state_dict = torch.load(file_path, weights_only=False, map_location=torch.device('cpu'))
        if isinstance(state_dict, nn.Module):
            print(f"{file_path} is a pytorch model.")
            return True
        
        print(f"{file_path} is not a pytorch model")
        return False
    
    except pickle.UnpicklingError:
        print(f"{file_path} is not a valid pickle.")
        return False
    except Exception as e:
        print(f"An error occurred when trying to verify if {file_path}")
        traceback.print_exc()
        return False

def isPytorchFile(filename : str):
    """
    Returns a boolean indicating whether the file is a pytorch model inference file.
    """
    pattern = r"^pytorch_model(?:-\d{5}-of-\d{5})?(?:\.[a-zA-Z0-9]+)?\.bin$"
    matches = re.match(pattern, filename)
    if matches:
        return True
    return False

def isValidFile(filename : str, mode : str):
    """
    Returns a boolean indicating the validity of a file for downloading based on
    whether it is in generalized mode or pytorch models only mode.
    """
    match(mode):
        case "generalized":
            for ext in SUPPORTED_FORMATS:
                if filename.endswith(ext):
                    return True
            return False

        case "pytorch":
            return isPytorchFile(filename)
        
        case "pytorch_model.bin":
            return filename == "pytorch_model.bin"

        case _:
            print("Invalid mode specified. Use 'generalized' or 'pytorch'.")
            sys.exit(1)


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
                print(f"Could not extract pickle files from {model_id}. Removing the folder")
                shutil.rmtree(local_dir)

        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")

    print(f"\nSuccessfully downloaded and processed {download_count} models.")

def download_models_with_space_limit(download_dir, space_limit_bytes, TAG, explored_models, download_log_df, limit, mode, last_success_download=None, direction="desc", mal_run=False):
    """
    Download models within the specified space limit
    """
    models = model_cache.getModelList(TAG, "likes", direction)
    
    downloaded_models = []
    current_size = 0
    start = 0
    i = 0
    security_infos = defaultdict(dict) # Store security info for downloaded models

    # Resuming from last successful download
    if last_success_download:
        last_download = last_success_download
        for model in models:
            i += 1 
            if model.modelId == last_download:
                start = i
                break
    
    for j in range(start, len(models)):
        model_id = models[j].modelId
        if len(set(download_log_df["name"])) >= limit:
            print(f"{limit} models have been downloaded")
            break
        
        # If it is a wild run, we are going in the opposite direction
        # If an already explored model is encountered, we break the loop
        # since it means we are moving into the benign set the model is trained on
        if model_id in explored_models and direction == "asc":
            print("Encountered an already explored model in the wild run. Stopping.")
            break
        
        i += 1
        try:
            print(f"\nChecking model: {model_id} in index {i + 1}")
            time.sleep(random.uniform(0.2, 0.75))  # Random delay to avoid rate limiting
            info = api.model_info(model_id)
            siblings = info.siblings

            # Shortlisting files from repo based on mode
            explored_name_files = set(zip(download_log_df['name'], download_log_df["filename"]))
            pkl_files = []
            for file in siblings:
                fname = file.rfilename.lower()
                if isValidFile(fname, mode):
                    if (model_id, fname) in explored_name_files:
                        print("Skipping already explored model.")
                        break
                    pkl_files.append(fname)
        
            if not pkl_files:
                print("No files with the supported format found. Skipping this model.")
                continue
            
            # Avoiding large/sharded models
            skip = False
            total_size_mb = 0
            for file in pkl_files:
                time.sleep(random.uniform(0.75, 1.5)) # Avoiding rate limits
                file_size = get_remote_file_size(model_id, file)
                if file_size is None:
                    print(f"Skipping model. Could not determine size for {file}")
                    skip = True
                    break
                
                file_size_mb = file_size / (1024 * 1024)
                total_size_mb += file_size_mb

            if skip:
                continue

            print(f"Total size of all files: {total_size_mb:.2f} MB")
            is_safe = check_memory_safety(total_size_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5) # Checking for possible OOM errors

            if not is_safe:
                print(f"Memory safety check failed for {model_id} with size {total_size_mb:.2f}MB. Skipping download.")
                continue
            
            if current_size + total_size_mb > space_limit_bytes:
                print(f"Adding {model_id} would exceed space limit. Stopping downloads.")
                break

            local_dir = os.path.join(download_dir, model_id.replace("/", "__"))    

            skip = False
            for file in pkl_files:
                print(f" - {file}")   
                pkl_file_path = os.path.join(local_dir, file)
                if os.path.exists(pkl_file_path):
                    print(f"Model {model_id} found in cache. Skipping download.")
                    actual_size = get_directory_size(local_dir)
                    current_size += actual_size
                    downloaded_models.append(model_id)
                    skip = True
                    break
            
            if skip:
                continue
            
            security_info = get_security_info(pkl_files, model_id, TAG)
            if mal_run:
                for fname in pkl_files:
                    if security_info.get(fname, {}).get("status", "") == "safe":
                        print(f"Skipping safe file: {fname}")
                        continue
                    elif security_info.get(fname, {}).get("status", "") == "unsafe":
                        print(f"Found unsafe file: {fname} with score {security_info[fname].get('score', 0)}") 
                        security_infos[model_id][fname] = security_info[fname].get('score', 0)
                    else:
                        print(f"Model file {fname} has no security info. Skipping.")
                        continue
            else:        
                if security_info.get(fname, {}).get("status", "") == "unsafe":
                    print(f"Found unsafe file: {fname} with score {security_info[fname].get('score', 0)}. Skipping.") # Skipping malicious
                    security_infos[model_id][fname] = security_info[fname].get('score', 0)
                    continue

            pkls = []
            # ckt_names = get_checkpoint_names(model_id, local_dir)
            for fname in pkl_files:
                time.sleep(random.uniform(0.2, 0.75))
                print(f"Downloading {fname}...")
                snapshot_download(
                    repo_id=model_id,
                    allow_patterns=[fname],
                    local_dir=local_dir,
                )
                
                pkl_file_path = os.path.join(local_dir, fname)
                if fname.endswith(".bin") or fname.endswith(".pt") or fname.endswith(".pth"):
                    # Check if it ispickle files
                    pkl = isPickle(pkl_file_path)
                elif fname.endswith(".pkl") or fname.endswith(".pickle"):
                    pkl = True
                else:
                    pkl = False

                # alternative handling of pickle files
                # if str(Path(pkl_file_path).relative_to(local_dir)) in ckt_names:
                #     print(f"{pkl_file_path} is a sharded model file (guaranteed pickle and pytorch model). Skipping Check.")
                #     pkl = True
                # else:
                #     pkl = isPickle_improved(pkl_file_path)

                if not pkl:
                    print(f"{fname} does not contain pickle files. Removing the file.")
                    os.remove(pkl_file_path)
                else:
                    pkls.append(fname)
                    
            if pkls:
                for fname in pkls:
                    actual_size = get_directory_size(local_dir)
                    current_size += actual_size
                    download_log_df.loc[len(download_log_df)] = {
                                                                    "name": model_id, 
                                                                    "likes": info.likes, 
                                                                    "downloads": info.downloads, 
                                                                    "last_updated": info.last_modified,
                                                                    "filename": fname, 
                                                                    "protectAiScan": security_info.get(fname, {}).get('score', {}).get("protectAiScan", 0), 
                                                                    "avScan": security_info.get(fname, {}).get('score', {}).get("avScan", 0),
                                                                    "pickleImportScan": security_info.get(fname, {}).get('score', {}).get("pickleImportScan", 0), 
                                                                    "jFrogScan": security_info.get(fname, {}).get('score', {}).get("jFrogScan", 0)
                                                                }

                    download_log_file_name = f"wild_{TAG}_downloaded_models.csv" if (direction == "asc" and not mal_run) else f"{TAG}_downloaded_models.csv"
                downloaded_models.append(model_id)
                save_model_metadata([download_log_df], [os.path.join(os.path.dirname(download_dir), download_log_file_name)])
                print(f"Successfully processed {model_id}")
                print(f"   - Actual size: {actual_size / (1024*1024):.2f} MB")
                print(f"   - Total size so far: {current_size / (1024*1024*1024):.2f} GB")
                print(f"Saved CSV file containing model metadata")
                # break # For retrieving malicious models only, comment otherwise to prevent large amount of requests to gcp
            else:
                print(f"Could not extract pickle files from {model_id}. Removing the folder")
                shutil.rmtree(local_dir)
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")
            traceback.print_exc()

    return downloaded_models, i, security_infos

def download_from_old_list(model_id, download_dir, space_limit_bytes, TAG, download_log_df, mode, last_success_download=None, mal_run=False):
    """
    Download a particular model from the old download list
    """
    downloaded_models = []
    current_size = 0
    i = 0
    security_infos = defaultdict(dict) # Store security info for downloaded models
    
    i += 1
    try:
        print(f"\nChecking model: {model_id} in index {i + 1}")
        time.sleep(random.uniform(0.2, 0.75))  # Random delay to avoid rate limiting
        info = api.model_info(model_id)
        siblings = info.siblings

        # Shortlisting files from repo based on mode
        explored_name_files = set(zip(download_log_df['name'], download_log_df["filename"]))
        pkl_files = []
        for file in siblings:
            fname = file.rfilename.lower()
            if isValidFile(fname, mode):
                if (model_id, fname) in explored_name_files:
                    print("Skipping already explored model.")
                    break
                pkl_files.append(fname)
    
        if not pkl_files:
            print("No files with the supported format found. Skipping this model.")
            return None, None, None

        # Avoiding large/sharded models
        skip = False
        total_size_mb = 0
        for file in pkl_files:
            time.sleep(random.uniform(0.75, 1.5)) # Avoiding rate limits
            file_size = get_remote_file_size(model_id, file)
            if file_size is None:
                print(f"Skipping model. Could not determine size for {file}")
                skip = True
                break
            
            file_size_mb = file_size / (1024 * 1024)
            total_size_mb += file_size_mb

        if skip:
            return None, None, None

        print(f"Total size of all files: {total_size_mb:.2f} MB")
        is_safe = check_memory_safety(total_size_mb, max_memory_gb=32, safety_margin_gb=2, safety_factor=1.5) # Checking for possible OOM errors

        if not is_safe:
            print(f"Memory safety check failed for {model_id} with size {total_size_mb:.2f}MB. Skipping download.")
            return None, None, None
        
        if current_size + total_size_mb > space_limit_bytes:
            print(f"Adding {model_id} would exceed space limit. Stopping downloads.")
            return None, None, None

        local_dir = os.path.join(download_dir, model_id.replace("/", "__"))    

        skip = False
        for file in pkl_files:
            print(f" - {file}")   
            pkl_file_path = os.path.join(local_dir, file)
            if os.path.exists(pkl_file_path):
                print(f"Model {model_id} found in cache. Skipping download.")
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                downloaded_models.append(model_id)
                skip = True
                break
        
        if skip:
            return None, None, None
        
        security_info = get_security_info(pkl_files, model_id, TAG)
        if mal_run:
            for fname in pkl_files:
                if security_info.get(fname, {}).get("status", "") == "safe":
                    print(f"Skipping safe file: {fname}")
                    continue
                elif security_info.get(fname, {}).get("status", "") == "unsafe":
                    print(f"Found unsafe file: {fname} with score {security_info[fname].get('score', 0)}") 
                    security_infos[model_id][fname] = security_info[fname].get('score', 0)
                else:
                    print(f"Model file {fname} has no security info. Skipping.")
                    continue
        
        pkls = []
        # ckt_names = get_checkpoint_names(model_id, local_dir)
        for fname in pkl_files:
            time.sleep(random.uniform(0.2, 0.75))
            print(f"Downloading {fname}...")
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[fname],
                local_dir=local_dir,
            )
            
            pkl_file_path = os.path.join(local_dir, fname)
            if fname.endswith(".bin") or fname.endswith(".pt") or fname.endswith(".pth"):
                # Check if it ispickle files
                pkl = isPickle(pkl_file_path)
            elif fname.endswith(".pkl") or fname.endswith(".pickle"):
                pkl = True
            else:
                pkl = False

            # alternative handling of pickle files
            # if str(Path(pkl_file_path).relative_to(local_dir)) in ckt_names:
            #     print(f"{pkl_file_path} is a sharded model file (guaranteed pickle and pytorch model). Skipping Check.")
            #     pkl = True
            # else:
            #     pkl = isPickle_improved(pkl_file_path)

            if not pkl:
                print(f"{fname} does not contain pickle files. Removing the file.")
                os.remove(pkl_file_path)
            else:
                pkls.append(fname)
                
        if pkls:
            for fname in pkls:
                actual_size = get_directory_size(local_dir)
                current_size += actual_size
                download_log_df.loc[len(download_log_df)] = {
                                                                "name": model_id, 
                                                                "likes": info.likes, 
                                                                "downloads": info.downloads, 
                                                                "last_updated": info.last_modified,
                                                                "filename": fname, 
                                                                "protectAiScan": security_info.get(fname, {}).get('score', {}).get("protectAiScan", 0), 
                                                                "avScan": security_info.get(fname, {}).get('score', {}).get("avScan", 0),
                                                                "pickleImportScan": security_info.get(fname, {}).get('score', {}).get("pickleImportScan", 0), 
                                                                "jFrogScan": security_info.get(fname, {}).get('score', {}).get("jFrogScan", 0)
                                                            }

                download_log_file_name = f"{TAG}_downloaded_models.csv"
            downloaded_models.append(model_id)
            save_model_metadata([download_log_df], [os.path.join(os.path.dirname(download_dir), download_log_file_name)])
            print(f"Successfully processed {model_id}")
            print(f"   - Actual size: {actual_size / (1024*1024):.2f} MB")
            print(f"   - Total size so far: {current_size / (1024*1024*1024):.2f} GB")
            print(f"Saved CSV file containing model metadata")
            # break # For retrieving malicious models only, comment otherwise to prevent large amount of requests to gcp
        else:
            print(f"Could not extract pickle files from {model_id}. Removing the folder")
            shutil.rmtree(local_dir)
    except Exception as e:
        print(f"Skipping {model_id} due to error: {e}")
        traceback.print_exc()

    return downloaded_models, i, security_infos

# Retrieve the list of models which need to be traversed and then check the security status of the models
# If it is unsafe, store it inside wild_malicious_models_api.csv
# If it is safe, then skip it.
def retrieve_mal_model_ids(limit, explored_models, direction, download_log_df, download_log_path, tag, last_success_download=None):
    """
    Retrieves the model IDs of the malicious models from the Hugging Face Hub detected as unsafe by the tools.
    """
    models = list(api.list_models(filter=tag, sort="likes"))
    if direction == "asc":
        models = list(reversed(models))
    print(f"Found {len(models)} models with tag '{tag}'")
    
    downloaded_models = []
    start = 0

    # Resuming from last successful download
    if last_success_download:
        last_download = last_success_download
        for model in models:
            start += 1 
            if model.modelId == last_download:
                break
    for j in range(start, len(models)):
        model_id = models[j].modelId
        if len(explored_models) + len(downloaded_models) >= limit:
            print(f"{limit} models have been downloaded")
            break
        
        if model_id in explored_models:
            print(f"Skipping already explored model: {model_id}")
            continue
        
        try:
            print(f"\nChecking model: {model_id} in index {j + 1}")
            time.sleep(random.uniform(0.5, 1.5))  # Random delay to avoid rate limiting

            info = api.model_info(model_id)
            siblings = info.siblings

            pkl_files = []
            for file in siblings:
                fname = file.rfilename.lower()
                for ext in SUPPORTED_FORMATS:
                    if fname.endswith(ext):
                        pkl_files.append(fname)
        
            if not pkl_files:
                print("No files with the supported format found. Skipping this model.")
                continue

            # Calculate actual size after extraction
            security_info = get_security_info(pkl_files, model_id, tag)
            mal_files = []
            for fname in pkl_files:
                if security_info.get(fname, {}).get("status", "") == "safe":
                    print(f"Skipping safe file: {fname}")
                    continue
                elif security_info.get(fname, {}).get("status", "") == "unsafe":
                    print(f"Found unsafe file: {fname} with score {security_info[fname].get('score', 0)}") 
                    mal_files.append(fname)
                else:
                    print(f"Model file {fname} has no security info. Skipping.")
                    continue

            if not mal_files:
                print(f"No malicious files found for model {model_id}. Skipping.")
                continue

            for fname in mal_files:
                security_score = security_info[fname].get('score', {})
                download_log_df.loc[len(download_log_df)] = {"name": model_id, 
                                                            "filename": fname,
                                                            "likes": info.likes, "downloads": info.downloads, 
                                                            "last_updated": info.last_modified, 
                                                            "protectAiScan": security_score.get("protectAiScan", 0), 
                                                            "avScan": security_score.get("avScan", 0), 
                                                            "pickleImportScan": security_score.get("pickleImportScan", 0), 
                                                            "jFrogScan": security_score.get("jFrogScan", 0),
                                                            "virusTotalScan": security_score.get("virusTotalScan", 0)
                                                            }
                downloaded_models.append(model_id)
                save_model_metadata([download_log_df], [download_log_path])
            print(f"Successfully processed {model_id}")
            print(f"Saved CSV file containing model metadata")
                
        except Exception as e:
            print(f"Skipping {model_id} due to error: {e}")
            print(traceback.format_exc())

    print(f"\nSuccessfully downloaded and processed {len(downloaded_models)} models.")

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
                    size_bytes = get_remote_file_size(dataset_id, fname, repo_type="dataset")
                    if size_bytes is None:
                        continue
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"Found file: {fname} - Size: {size_mb:.2f} MB")
                    if size_mb <= SPACE_LIMIT_GB * 1024:  
                        filtered_datasets.append((dataset_id, fname, size_mb))
                        print(f"Added: {dataset_id}/{fname} ({size_mb:.2f} MB)")
                        break

            if len(filtered_datasets) >= LIMIT_MATCHES:
                break

        except Exception as e:
            print(f"Skipping {dataset_id} due to error: {e}")
            continue

    print(f"\n{len(filtered_datasets)} datasets passed the filter.")

    for dataset_id, filename, size_mb in filtered_datasets:
        print(f"Downloading {dataset_id}/{filename} ({size_mb:.2f} MB)...")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            allow_patterns=[filename],
            local_dir=os.path.join(DOWNLOAD_DIR, "dataset",dataset_id.replace("/", "__")),
            local_dir_use_symlinks=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models or datasets from Hugging Face Hub.")
    parser.add_argument(
        "--type",
        choices=["model", "dataset"],
        required=True,
        help="Specify what to download: 'model' or 'dataset'"
    )
    parser.add_argument(
        "--tag",
        default=TAG,
        required=True,
        help="Tag to filter models or datasets (default: 'text-generation-inference')"
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="Tag to find how much space LIMIT_MATCHES models or datasets would take if downloaded"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of models to download (default: 50)",
    )
    args = parser.parse_args()

    TAG = args.tag

    # download_dir = os.path.join("data", "malicious_dataset", "model", TAG)
    download_log = os.path.join("data", "clean_dataset", "model", f"{TAG}_downloaded_models_copy.csv")
    wild_mal_log = os.path.join(os.path.dirname(download_log), f"wild_malicious_models_{TAG}_api.csv")
    wild_mal_models = get_model_metadata(wild_mal_log, ["name", "filename", "likes", "downloads", "last_updated", "protectAiScan", "avScan", "pickleImportScan", "jFrogScan", "virusTotalScan", "clf_prediction", "clf_decision_score"])

    retrieve_mal_model_ids(
        limit=args.limit,
        explored_models=set(wild_mal_models["name"].tolist() if not wild_mal_models.empty else []),
        direction="asc",
        download_log_df=wild_mal_models,
        download_log_path=wild_mal_log,
        last_success_download="xfey/Qwen2.5-7B-Whitebox-GSM8k-Exp005",
        tag=TAG
    )