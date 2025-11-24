import pandas as pd
import os

UNQ_SYS_CALLS = {"fchdir", "clone", "wait4"}  # unique system calls to malicious model files(pickles)
DATA_DIR = "classifier/data"  # directory where the system call data would be present

mal_counts = pd.read_csv(os.path.join(DATA_DIR, "malicious_syscall_counts_test.csv"))

def is_malicious(row):
    for syscall in UNQ_SYS_CALLS:
        if syscall in row and pd.notna(row[syscall]) and float(row[syscall]) >= 1:
            return True
        elif syscall == "execve" and float(row[syscall]) > 1:
            return True
    return False

mal_counts['detected_malicious'] = mal_counts.apply(is_malicious, axis=1)

print(mal_counts[["name", "detected_malicious"]])

detected_count = mal_counts['detected_malicious'].sum()
total_files = len(mal_counts)

precision = (detected_count / detected_count) if detected_count > 0 else 0 # Since this dataset is confirmed to be malicious, false positives is impossible and true positives would be all the detections

print(f"Detected malicious files: {detected_count}/{total_files}")
print(f"Precision: {precision:.2f}")