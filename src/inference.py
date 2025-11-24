import torch 
import traceback
from huggingface_hub import hf_hub_download
from utils.utils import extract_model_id
import argparse
import os
import psutil
import pickle
import sys

MODEL_DIR = "./data/malicious_dataset/model"  # Directory where the models are stored

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    # s = socket.socket()
    # s.bind(("0.0.0.0", 5000))
    # s.listen(1)

    parser = argparse.ArgumentParser(description="Run inference on a Hugging Face repository.")
    parser.add_argument("--repo_name", type=str, help="The name of the Hugging Face repository.")
    parser.add_argument("--pkl_file", type=str, help="The name of the pickle file to load.")
    parser.add_argument("--online_mode", type=str, choices=["true", "false"], help="Whether to run in online mode (download from Hugging Face) or offline mode (use local files).", default="false")
    parser.add_argument("--clean_mode", type=str, choices=["true", "false"], help="Whether to run models of clean dataset.", required=False, default="false")
    args = parser.parse_args()
    
    base_dir = os.getcwd()
    repo_dir = None
    if args.online_mode.lower() == "true":
        # Download all the bin and pkl files to local storage
        repo_dir = hf_hub_download(repo_id=args.repo_name, filename=args.pkl_file)
    else:
        if args.clean_mode.lower() == "true":
            # use the clean dataset version of the models for the pkl file
            # repo_dir = f"{CLEAN_DATASET}/{args.repo_name}/{args.pkl_file}"
            repo_dir = os.path.join(base_dir, args.pkl_file)
            if not os.path.exists(repo_dir):
                print(f"Error: The specified clean dataset file {repo_dir} does not exist.")
                return
        else:
            # use the zenodo version of the models for the pkl file
            # Specifically harcoded to the zenodo dataset
            model_name = extract_model_id(args.repo_name)
            if not model_name:
                print(f"Error: Invalid repository name {args.repo_name}.")
                return
            # repo_dir = f"{MODEL_DIR}/{args.repo_name}/{model_name}/{args.pkl_file}"
            repo_dir = os.path.join(MODEL_DIR, args.repo_name, model_name, args.pkl_file)
    
    try:
        print(f"Memory usage before deserialization: {get_memory_usage()}MB")
        original_stdout = sys.stdout
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1, closefd=False)
        with open(f"{repo_dir}", "rb") as f:
            with torch.no_grad():
                torch.load(f, weights_only=False, map_location=torch.device('cpu'))
        sys.stdout = original_stdout
        print(f"Memory usage after deserialization: {get_memory_usage()}MB")
    except Exception as e:
        print(f"an exception occurred for {args.repo_name} : {e}")
        traceback.print_exc()
        sys.exit(1)
    except pickle.UnpicklingError as e:
        print(f"pickle.UnpicklingError occurred. This file may be corrupted or not a pickle file at all: {args.pkl_file}")
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()