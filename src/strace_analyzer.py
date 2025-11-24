"""
Strace System Call Analysis Script
Analyzes system call frequencies between malicious and benign pickle files
"""

import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
from utils.utils import get_repo_and_file_name
import traceback
from collections import defaultdict
import argparse

# Putting the hybrid models to training
# Evaluating the already trained models on the new malhug

# KEY_PATTERNS = [
#     # Process execution and control
#     'execve', 'fork', 'clone', 'clone3', 'vfork', 'ptrace',
    
#     # Network operations  
#     'socket', 'connect', 'bind', 'accept', 'listen', 
#     'sendto', 'recvfrom', 'sendmsg', 'recvmsg',
    
#     # File operations
#     'open', 'openat', 'creat', 'unlink', 'unlinkat', 'rename', 'renameat',
#     'write', 'read', 'access', 'faccessat', 'chmod', 'chown',
    
#     # Memory operations
#     'mmap', 'munmap', 'mremap',
    
#     # Directory operations
#     'mkdir', 'mkdirat', 'rmdir', 'chdir',
    
#     # Process control
#     'kill', 'tkill', 'prctl', 'setuid', 'setgid', 'reboot' 
# ]

CLASSIFIER_DIR = "classifier/data"
class StraceAnalyzer:
    def __init__(self, filter_syscalls=False):
        self.benign_name_data = [] # system call count data with name of repo
        self.malicious_name_data = []# system call count data with name of repo
        self.filtersyscalls = filter_syscalls # Filtering for sensitive/security-critical system calls
        
    def parse_strace_count(self, file_path):
        # """Parse strace output file and extract system call counts"""
        syscall_counts = {}
        with open(file_path, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        in_summary = False
        for line in lines:
            # Skip header, separator lines and footer
            if line.startswith('% time') or line.startswith('------') or line.startswith('100.00') or line.endswith('...>'):
                # in_summary = True
                continue
                
            if line.strip():
                parts = line.split()
                # if len(parts) >= 5:
                try:
                    calls = int(parts[3]) # Extracting the frequency of the calls
                    syscall = parts[-1]  # Last part is the syscall name
                    syscall_counts[syscall] = calls
                except (ValueError, IndexError):
                    print(f"Error parsing line: {line} for file {file_path}")
                    traceback.print_exc()
                    continue
                # else:
                #     print(parts)
        return syscall_counts
    

    def generate_ngram(self, seqs, n=2):
        """
        Generate n-grams from a sequence
        """
        ngrams = defaultdict(int)

        if len(seqs) < n:
            print("Sequence is too short for n-grams")
            return ngrams
        
        for i in range(len(seqs) - n + 1):
            ngram = tuple(seqs[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def assign_pname(self, pid, pid_map):
        """
        Assign a process name to a PID
        Example: P1, P2, ....
        """
        if pid not in pid_map:
            pid_map[pid] = f"P{len(pid_map) + 1}"
        return pid_map[pid]

    def parse_strace_seq(self, file_path, n=2):
        """
        Extracts system calls from strace log files and produce n-gram sequences
        """   
        syscalls = []
        pid_with_syscalls = []  # List to store tuples of (pid, syscall)
        pid_map = {}

        with open(file_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')

        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) < 2:
                    print(f"Skipping line with insufficient parts: {line}")
                    continue
                pid = parts[0]  
                syscall = parts[1]
                end_idx = syscall.find('(')
                if end_idx == -1: # No syscall name found
                    continue
                syscall_name = syscall[:end_idx]  # Extract syscall name before '('
                syscalls.append(syscall_name)
                pname = self.assign_pname(pid, pid_map)  # Assign process name to PID
                pid_with_syscalls.append((pname, syscall_name))

        if len(syscalls) < n or len(pid_with_syscalls) < n:
            print("Sequence is too short for n-grams")
            return None

        # Generate the n-gram sequences
        syscall_ngrams = self.generate_ngram(syscalls, n)
        pid_ngrams = self.generate_ngram(pid_with_syscalls, n)

        return syscall_ngrams, pid_ngrams

    def merge_features(self, syscall_counts, syscall_seq_counts, pid_with_syscall_seq_counts):
        """
        Merge different types of features into a coherent dictionary structure for each sample
        """
        # Check if the lengths match
        if not (len(syscall_counts) == len(syscall_seq_counts) == len(pid_with_syscall_seq_counts)):
            print(f"Warning: Input lists have different lengths: {len(syscall_counts)}, {len(syscall_seq_counts)}, {len(pid_with_syscall_seq_counts)}")
            # Take the minimum length to avoid index errors
            min_len = min(len(syscall_counts), len(syscall_seq_counts), len(pid_with_syscall_seq_counts))
            syscall_counts = syscall_counts[:min_len]
            syscall_seq_counts = syscall_seq_counts[:min_len]
            pid_with_syscall_seq_counts = pid_with_syscall_seq_counts[:min_len]
        
        merged_features = []
        
        for i in range(len(syscall_counts)):
            combined_dict = syscall_counts[i].copy()
            
            # Converting seq tuple into pandas interpretable format
            for seq_tuple, count in syscall_seq_counts[i].items():
                seq_key = "seq_" + "_".join(seq_tuple)
                combined_dict[seq_key] = count
            
            for proc_seq_tuple, count in pid_with_syscall_seq_counts[i].items():
                proc_parts = []
                for item in proc_seq_tuple:
                    proc_parts.append(f"{item[0]}:{item[1]}") # Converting to "PID:syscall"
                
                proc_seq_key = "proc_seq_" + "_".join(proc_parts)
                combined_dict[proc_seq_key] = count
                
            merged_features.append(combined_dict)
        
        return merged_features

    def load_data(self, malicious_dir="", benign_dir="", n=2):
        """
        Load all strace files from malicious and benign directories
        """
        
        if not malicious_dir and not benign_dir:
            print("Provide atleast a benign or malicious strace directory path.")
            return 

        # Load the count of system calls in malicious files
        if malicious_dir:
            syscall_freq = []
            mal_files_count = list(Path(malicious_dir).glob('*_count.log'))
            for file_path in mal_files_count:
                syscall_counts = self.parse_strace_count(file_path)
                if syscall_counts:
                    log_name = os.path.basename(file_path)
                    repo_name, filename = get_repo_and_file_name(log_name)
                    # repo_name = get_repo_and_file_name(log_name)
                    if repo_name:
                        syscall_counts["name"] = repo_name
                        syscall_counts["filename"] = filename
                    syscall_freq.append(syscall_counts)

            syscall_seq_freq = []
            pid_with_syscall_seq_freq = []
            # Load the sequence of system calls in malicious files
            mal_files_seq = list(Path(malicious_dir).glob('*_logs.log'))
            for file_path in mal_files_seq:
                mal_syscall_seqs, mal_pid_with_syscall_seqs = self.parse_strace_seq(file_path, n)
                if mal_syscall_seqs and mal_pid_with_syscall_seqs:
                    syscall_seq_freq.append(mal_syscall_seqs)
                    pid_with_syscall_seq_freq.append(mal_pid_with_syscall_seqs)

            # Merge features into one dictionary
            self.malicious_name_data = self.merge_features(syscall_freq, syscall_seq_freq, pid_with_syscall_seq_freq)
        
        # Load benign files
        if benign_dir:
            syscall_freq = []
            ben_files = list(Path(benign_dir).glob('*_count.log'))
            for file_path in ben_files:
                syscall_counts = self.parse_strace_count(file_path)
                if syscall_counts:
                    log_name = os.path.basename(file_path)
                    repo_name, filename = get_repo_and_file_name(log_name)
                    if repo_name:
                        syscall_counts["name"] = repo_name
                        syscall_counts["filename"] = filename
                    syscall_freq.append(syscall_counts)
            
            syscall_seq_freq = []
            pid_with_syscall_seq_freq = []
            ben_files_seq = list(Path(benign_dir).glob('*_logs.log'))
            for file_path in ben_files_seq:
                ben_syscall_seqs, ben_pid_with_syscalls_seqs = self.parse_strace_seq(file_path, n)
                if ben_syscall_seqs and ben_pid_with_syscalls_seqs:
                    syscall_seq_freq.append(ben_syscall_seqs)
                    pid_with_syscall_seq_freq.append(ben_pid_with_syscalls_seqs)
            # Merge features into one dictionary
            self.benign_name_data = self.merge_features(syscall_freq, syscall_seq_freq, pid_with_syscall_seq_freq)
        
            print(f"Loaded {len(self.malicious_name_data)} malicious samples")
            print(f"Loaded {len(self.benign_name_data)} benign samples")
        
    def ensure_order(self, tag : str, data : pd.DataFrame):
        """
        Orders the dataframe according to the order in which the models were downloaded.
        Takes reference from the downloaded csv created during the model download process.
        """
        ordered_file_path = f"data/clean_dataset/model/{tag}_downloaded_models.csv"
        if not os.path.exists(ordered_file_path):
            print(f"The ordered file does not exist. Please check if this file exists: {ordered_file_path}")
            return None
        ordered_df = pd.read_csv(ordered_file_path)
        ordered_names = ordered_df["name"]

        name_order = {name: idx for idx, name in enumerate(ordered_names)}
        
        data['_sort_idx'] = data['name'].map(name_order)
        
        data = data.sort_values('_sort_idx').drop('_sort_idx', axis=1)
        
        return data
    
    def save_count_per_sample(self, mode="normal_run", tag="all", ben=False, ext="csv"):
        """Save system call counts per sample to CSV or Parquet files files"""
        ben_counts = pd.DataFrame(self.benign_name_data)
        mal_counts = pd.DataFrame(self.malicious_name_data)

        if ben: # Save benign strace data

            if ben_counts.empty:
                print("Benign strace data is empty. Please check if you provided the right directory for the benign_dir parameter in the load_data function.")
                return
            
            save_dir = os.path.join(CLASSIFIER_DIR, f"{mode}_benign_syscall_counts_{tag}.{ext}")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            print("Ordering the entries")
            ben_counts = self.ensure_order(tag, ben_counts)
            print(f"Ordered the entries: {ben_counts['name']}")

            if ext == 'csv':
                ben_counts.to_csv(save_dir, index=False)
            elif ext == 'parquet':
                ben_counts.to_parquet(save_dir, index=False, engine='pyarrow')
            else: # Sanity check
                print("Invalid extension format. Please provide either csv or parquet.")
                return
            
        else: # Save malicious strace data

            if mal_counts.empty:
                print("Malicious strace data is empty. Please check if you provided the right directory for the malicious_dir parameter in the load_data function.")
                return
            
            save_dir = os.path.join(CLASSIFIER_DIR, f"{mode}_malicious_syscall_counts_{tag}.{ext}")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            
            if ext == 'csv':
                mal_counts.to_csv(save_dir, index=False)
            elif ext == 'parquet':
                mal_counts.to_parquet(save_dir, index=False, engine='pyarrow')
            else:
                print("Invalid extension format. Please provide either csv or parquet.")
                return            
        
        print(f"System call counts saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the strace logs collected from the pipeline in main.py to structure documents like csv or parquet")
    parser.add_argument('--tag', help='The tag of the PTMs of which the strace are analyzed.')
    parser.add_argument('--mode', help='An optional argument to give the csv/parquet a more descriptive name.')
    parser.add_argument('--benign', action='store_true', help='A boolean argument to decide whether a benign or malicious structured dataset should be saved from the analyzed logs. Presence of this argument means a benign structured dataset would be saved from the strace logs analyzed in the benign_dir. Absence means the vice versa.')
    parser.add_argument('--malicious-dir', help='The directory containing the traces of the malicious strace logs.')
    parser.add_argument('--benign-dir', help='The directory containing the traces of the benign strace logs.')
    parser.add_argument('--ext', choices=['csv', 'parquet'], help='Type of structured document to save the strace data in (csv or parquet).')

    args = parser.parse_args()
    analyzer = StraceAnalyzer()

    analyzer.load_data(malicious_dir=args.malicious_dir, benign_dir=args.benign_dir)

    analyzer.save_count_per_sample(tag=args.tag, mode=args.mode, ben=args.benign, ext=args.ext)