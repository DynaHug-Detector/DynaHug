from pathlib import Path
import zstandard
import tarfile
import os
from google.cloud import storage
import pandas as pd
import traceback
import re
import psutil
import platform
import subprocess
import resource
from huggingface_hub import HfApi
import hashlib


def parse_files(files):
    """
    Parsing the filenames from the dataframe(malhug_result_info.csv) 
    """
    res = []
    file_list = files.split(",") 
    for file in file_list:
        idx = file.find(":")
        res.append(file[:idx if idx != -1 else len(file)])  # Get the filename before the colon if it exists
    return res

def parse_feature_from_file(file_path):
    """
    Parses the list of all syscalls/opcodes from a file.
    """
    syscalls = []
    with open(file_path, 'r') as f:
        for line in f:
            syscall = line.strip()
            if syscall:  # Ensure the line is not empty
                syscalls.append(syscall)
    return syscalls

def extract_model_id(repo_name):
    """
    Extracts the model ID from the repo_name.
    """
    idx = repo_name.find("/")
    if idx != -1:
        return repo_name[idx + 1:]
    return None


def find_files_with_repo_names(ext, root_dir, model_id=None):
    """
    Returns a list of paths of all the files with extension 'ext' along with their corresponding repo names.
    This is for clean models
    """
    res = []
    root_path = Path(root_dir)

    for file in root_path.rglob(ext):
        # Find the repo name as the first directory under root_dir
        try:
            relative = file.relative_to(root_path)
            repo_name = relative.parts[0] if len(relative.parts) > 1 else None
            if repo_name:
                if model_id:
                    modified_relative = Path(*relative.parts[1:]) if len(relative.parts) > 1 else relative
                    res.append((model_id.replace("/", "__"), str(modified_relative), str(file)))
                else:
                    res.append((repo_name, str(relative), str(file)))
        except ValueError:
            continue
    return res

def remove_file_from_directory(dir, filename):
    """
    Removes a specific file from the specified directory
    """
    try:
        matches = Path(dir).glob(filename) # Only topmost directory deletion
        if matches:
            for target in matches:
                print(f"{target} found. Deleting...")
                target.unlink()
        else:
            print(f"{filename} not found in {dir}")
            
    except Exception as e:
        print(f"An exception occurred when trying to remove a file: {e}")

def get_all_syscalls(file_path):
    """
    Extracts all the system calls present inside a file and saves them in a text file.
    Designed to extract the system calls from either /usr/include/asm/unistd_32.h or /usr/include/x86_64-linux-gnu/asm/unistd_64.h.
    """
    with open(file_path, 'r') as f:
        syscalls = []
        for line in f:
            match = re.match(r'#define\s+__NR_([a-zA-Z0-9_]+)', line)
            if match:
                syscalls.append(match.group(1))
    
    with open('syscalls.txt', 'w') as out_file:
        for syscall in syscalls:
            out_file.write(f"{syscall}\n")

def print_system_info():
    """Print detailed information about the system environment"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # OS Information
    print(f"Operating System: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    
    # CPU Information
    cpu_info = {}
    try:
        if platform.system() == "Linux":
            # Get CPU model name on Linux
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["model"] = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":  # macOS
            cpu_info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception as e:
        cpu_info["model"] = "Unknown (Error retrieving CPU info)"
    
    print(f"CPU: {cpu_info.get('model', 'Unknown')}")
    print(f"CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    print(f"CPU Cores (Logical): {psutil.cpu_count(logical=True)}")
    
    # Memory Information
    svmem = psutil.virtual_memory()
    print(f"Total RAM: {svmem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {svmem.available / (1024**3):.2f} GB")
    print(f"Used RAM: {svmem.used / (1024**3):.2f} GB ({svmem.percent}%)")
    
    # Disk Information
    disk = psutil.disk_usage('/')
    print(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    print(f"Disk Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)")
    print(f"Disk Free: {disk.free / (1024**3):.2f} GB")
    
    # Network Information
    try:
        if_addrs = psutil.net_if_addrs()
        for interface_name, interface_addresses in if_addrs.items():
            for address in interface_addresses:
                if str(address.family) == 'AddressFamily.AF_INET':
                    print(f"Network Interface: {interface_name}, IP: {address.address}")
    except:
        print("Unable to retrieve network information")
    
    print("="*60 + "\n")

def get_model_name_from_logs(log_file_name):
    """
    Extracts the model name from the log file name.
    """
    match = re.match(r"strace_logs_(.+)_count\.log$", log_file_name)
    if match:
        return match.group(1).replace("__", "/")
    return None

def get_model_metadata_from_api(repo_name):
    """
    Fetches model metadata from the Hugging Face API.
    """
    api = HfApi()
    try:
        model_info = api.model_info(repo_name)
        metadata = {
            "name": repo_name,
            "likes": model_info.likes,
            "downloads": model_info.downloads,
            "last_modified": model_info.last_modified
        }
        return metadata
    except Exception as e:
        print(f"Error fetching metadata for {repo_name}: {e}")
        return {}

def create_analysis_archive(download_dir, output_archive, compression_level=3):
    """
    Create a zip archive containing downloaded models and analysis results
    """
    print(f"Creating archive: {output_archive}")

    compressor = zstandard.ZstdCompressor(level=compression_level)
    
    with open(output_archive, 'wb') as f:
        with compressor.stream_writer(f) as writer:
            with tarfile.open(fileobj=writer, mode='w|') as tar:
                tar.add(download_dir, arcname=os.path.basename(download_dir))

def extract_analysis_archive(archive_path, extract_dir):
    """
    Extract a zstd compressed tar archive
    """
    print(f"Extracting archive: {archive_path} to {extract_dir}")

    decompressor = zstandard.ZstdDecompressor()
    try:
        with open(archive_path, 'rb') as f:
            with decompressor.stream_reader(f) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    tar.extractall(path=extract_dir)

        return True
    except Exception as e:
        print("An error occurred, stopped extracting")
        traceback.print_exc()
        return False

def get_folders_from_directory(target_dir):
    """
    Extract folder names from a specific directory.
    """
    try:
        # List all items in the directory
        items = os.listdir(target_dir)
        
        # Filter only directories
        folders = [item for item in items if os.path.isdir(os.path.join(target_dir, item))]
        
        return sorted(folders)
    
    except FileNotFoundError:
        print(f"Directory '{target_dir}' not found")
        return []
    except PermissionError:
        print(f"Permission denied accessing '{target_dir}'")
        return []
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

def upload_to_gcs(file_path, bucket_name, blob_name):
    """
    Upload a file to Google Cloud Storage
    """
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

        print(f"Downloading {blob_name} from GCP bucket {bucket_name} to {destination_file_name}")
        blob.download_to_filename(destination_file_name)
        
        print(f"Successfully downloaded to {destination_file_name}")
        return True
        
    except Exception as e:
        print(f"Error downloading from GCS: {e}")
        return False

def calculate_folder_checksum(folder_path):
    """Calculate SHA256 checksum for pytorch_model.bin file in a folder."""
    hasher = hashlib.sha256()
    model_file = os.path.join(folder_path, "pytorch_model.bin")
    
    try:
        if not os.path.isfile(model_file):
            print(f"  Warning: pytorch_model.bin not found in {folder_path}")
            return None
        
        with open(model_file, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating checksum for {model_file}: {e}")
        return None

def validate_folders(parent_directory, checklist):
    """
    Check folders against a checklist and calculate checksums for matches.
    """
    results = {}
    
    if not os.path.isdir(parent_directory):
        print(f"Error: {parent_directory} is not a valid directory")
        return results
    
    try:
        for item in os.listdir(parent_directory):
            item_path = os.path.join(parent_directory, item)
            
            if os.path.isdir(item_path) and item in checklist:
                print(f"Processing: {item}")
                checksum = calculate_folder_checksum(item_path)
                if checksum:
                    results[item] = checksum
                    print(f"  Checksum: {checksum}")
                else:
                    print(f"  Failed to calculate checksum")
    
    except PermissionError:
        print(f"Error: Permission denied accessing {parent_directory}")
    except Exception as e:
        print(f"Error: {e}")
    
    return results


def get_cpu_time():
    """Get total CPU time (user + system) in seconds"""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def get_model_metadata(file_path, columns):
    """
    Getting dataframe of the model id along with its metadata
    """
    df = pd.DataFrame(columns=columns)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    return df


def save_model_metadata(dfs, file_paths):
    """
    Save the metadata collected of all the models which are downloaded for the dataset.
    """
    try:
        for df, path in zip(dfs, file_paths):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Saved metadata to {path}")
    except Exception:
        print("An error occurred while saving the csv")
        traceback.print_exc()

def get_repo_and_file_name(log_file_name):
    """
    Extracts the model name from the log file name.
    """
    match = re.match(r"strace_logs_(.+)--(.+?)_count\.log$", log_file_name)
    # match = re.match(r"strace_logs_(.+)_count\.log$", log_file_name)
    if match:
        repo_name = match.group(1).replace("__", "/", 1)
        filename = match.group(2).replace("__", "/")
        return repo_name, filename
        # return repo_name
    return None, None

if __name__ == "__main__":
    # ls = find_files_with_repo_names("*.bin", "data/clean_dataset/model") + find_files_with_repo_names("*.pkl", "data/clean_dataset/model")
    # print(len(ls))
    # print(ls)
    # upload_to_gcs("output.log",  "-model-archive", "test/output.log")
    # create_analysis_archive("data/clean_dataset/model/text-generation", "analysis_result.tar.zst")
    # res = list_files_gcs("-model-archive", prefix="analysis_results/text-generation/")
    # print(res)
    # get_all_syscalls("/usr/include/x86_64-linux-gnu/asm/unistd_64.h")

    download_from_gcs("-crawling-archive", "analysis_results/text-generation/analysis_archive_text-generation_batch_1341.tar.zst", "data/clean_dataset/model/text-generation/analysis_result.tar.zst")

    # extract_analysis_archive("data/malicious_dataset/model/text-generation/analysis_result.tar.zst", "data/malicious_dataset/model/")

    # res = validate_folders("data/malicious_dataset/model/text-generation", ["Branis333__astro-gpt2-chatbot"])

    # print(res)