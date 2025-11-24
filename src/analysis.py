import subprocess
import os
import traceback
import gc

OUTPUT_DIR = "output" # Directory where the output of dynamic analysis will be saved
CLEAN_DATASET = "data/clean_dataset" # Folder containing clean dataset files
MALICIOUS_DATASET = "data/malicious_dataset" # Folder containing malicious dataset files
TIMEOUT = 120 # Timeout for strace commands in seconds

"""
Runs strace in the main thread and writes the output to a file.
"""
def run_strace(repo_name, output_file, out_dir, rel_path, file_name, isPickle=True, isClean=False):
    # Getting absolute path to the virtual environment
    # project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.getcwd()
    virt_python_path = os.path.join(project_root, ".venv", "bin", "python") 

    stdout1, stderr1 = None, None
    stdout2, stderr2 = None, None
    returncode1, returncode2 = None, None
    exception_info = None
    os.makedirs(out_dir, exist_ok=True)

    gc.collect()
    with open(os.path.join(out_dir, f"dynamic_analysis_{repo_name.replace('/', '-')}_{rel_path}.log"), "w") as f:
        if isPickle:
            f.write(f"Pickle deserialization output for {repo_name} with pkl file {file_name}:\n")
            try:
                f.write(f"------------- For Run 1(strace logs) -------------\n")
                print(f"Running strace for {repo_name} with pkl file {file_name}")
                command_logs = (
                    f"timeout --preserve-status --signal=TERM {TIMEOUT}s strace -f -o {output_file}_logs.log {virt_python_path} src/inference.py --repo_name {repo_name} --pkl_file {file_name} --online_mode false --clean_mode {str(isClean).lower()}"
                )

                # res1 = subprocess.Popen(
                #     command_logs,
                #     capture_output=True, text=True, timeout=15, shell=True
                # )
                
                # gc.collect()
                # process = subprocess.Popen(
                #     command_logs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
                # )
                # stdout1, stderr1 = process.communicate()  # Wait for the command to complete
                # returncode1 = process.returncode
                # gc.collect()

                gc.collect()
                result1 = subprocess.run(
                    command_logs,
                    capture_output=True,
                    text=True,
                    shell=True
                )
                gc.collect()
                stdout1, stderr1 = result1.stdout, result1.stderr
                returncode1 = result1.returncode
            except subprocess.TimeoutExpired as timeErr:
                # process.terminate()
                exception_info = traceback.format_exc()
                f.write(f"Timeout expired for {repo_name} with pkl file {file_name}:\n")
                captured_output = timeErr.stdout.decode('utf-8', errors='ignore') if timeErr.stdout else ""
                captured_error = timeErr.stderr.decode('utf-8', errors='ignore') if timeErr.stderr else ""
                f.write(f"Captured output: {captured_output}\n")
                f.write(f"Captured error: {captured_error}\n")
            finally: 
                if stdout1 is not None and stderr1 is not None:
                    f.write(stdout1)
                    f.write(stderr1)
            gc.collect()
            try:
                f.write("\n\n")
                f.write(f"------------- For Run 2(strace count) -------------\n")
                command_count = (
                    f"timeout --preserve-status --signal=TERM {TIMEOUT}s strace -c -f -o {output_file}_count.log {virt_python_path} src/inference.py --repo_name {repo_name} --pkl_file {file_name} --online_mode false --clean_mode {str(isClean).lower()}"
                )

                # gc.collect()
                # process = subprocess.Popen(
                #     command_count, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
                # )
                # stdout2, stderr2 = process.communicate()  # Wait for the command to complete
                # returncode2 = process.returncode

                # gc.collect()
                gc.collect()
                result2 = subprocess.run(
                    command_count,
                    capture_output=True,
                    text=True,
                    shell=True,
                )

                gc.collect()
                stdout2, stderr2 = result2.stdout, result2.stderr
                returncode2 = result2.returncode
            except subprocess.TimeoutExpired as timeErr:
                # process.terminate()
                exception_info = traceback.format_exc()
                f.write(f"Timeout expired for {repo_name} with pkl file {file_name}:\n")
                captured_output = timeErr.stdout.decode('utf-8', errors='ignore') if timeErr.stdout else ""
                captured_error = timeErr.stderr.decode('utf-8', errors='ignore') if timeErr.stderr else ""
                f.write(f"Captured output: {captured_output}\n")
                f.write(f"Captured error: {captured_error}\n")
            finally:
                if stdout2 is not None and stderr2 is not None:
                    f.write(stdout2)
                    f.write(stderr2)
                f.write("\n\n")
                if exception_info:  
                    f.write("Exception occurred during strace execution:\n")
                    f.write(exception_info)
            gc.collect()
        else:
            f.write(f"Dataset loading script output for {repo_name} with dataloading script {file_name}:\n")
            try:
                f.write(f"------------- For Run 1(strace logs) -------------\n")
                print(f"Running strace for {repo_name} with dataloading script {file_name}")
                command_logs = None
                if isClean:
                    command_logs = (
                        f"strace -f -o {output_file}_logs.log {virt_python_path} '{file_name}'"
                    )
                else:
                    # For malicious dataset, we need to run the dataloading script directly
                    command_logs = (
                        f"strace -f -o {output_file}_logs.log {virt_python_path} '{MALICIOUS_DATASET}/dataset/{repo_name}/{file_name}'"
                    )
                res1 = subprocess.run(
                    command_logs,
                    capture_output=True, text=True, timeout=15, shell=True
                )
            except subprocess.TimeoutExpired as timeErr:
                exception_info = traceback.format_exc()
                f.write(f"Timeout expired for {repo_name} with dataloading script {file_name}:\n")
                captured_output = timeErr.stdout.decode('utf-8', errors='ignore') if timeErr.stdout else ""
                captured_error = timeErr.stderr.decode('utf-8', errors='ignore') if timeErr.stderr else ""
                f.write(f"Captured output: {captured_output}\n")
                f.write(f"Captured error: {captured_error}\n")
                
            finally:
                if res1:
                    f.write(res1.stdout)
                    f.write(res1.stderr)
            try:
                f.write("\n\n")
                f.write(f"------------- For Run 2(strace count) -------------\n")
                command_count = None
                if isClean:
                    command_count = (
                        f"strace -c -f -o {output_file}_count.log {virt_python_path} '{file_name}'"
                    )
                else:
                    command_count = (
                        f"strace -c -f -o {output_file}_count.log {virt_python_path} '{MALICIOUS_DATASET}/dataset/{repo_name}/{file_name}'"
                    )
                res2 = subprocess.run(
                    command_count,
                    capture_output=True, text=True, timeout=15, shell=True
                )
            except subprocess.TimeoutExpired as timeErr:    
                exception_info = traceback.format_exc()
                f.write(f"Timeout expired for {repo_name} with dataloading script {file_name}:\n")
                captured_output = timeErr.stdout.decode('utf-8', errors='ignore') if timeErr.stdout else ""
                captured_error = timeErr.stderr.decode('utf-8', errors='ignore') if timeErr.stderr else ""
                f.write(f"Captured output: {captured_output}\n")
                f.write(f"Captured error: {captured_error}\n")
            finally:
                if res2:
                    f.write(res2.stdout)
                    f.write(res2.stderr)
                f.write("\n\n")
                if exception_info:
                    f.write("Exception occurred during strace execution:\n")
                    f.write(exception_info)
    return returncode1 == 0 and returncode2 == 0  # Return True if both strace commands were successful

"""
To run the main dynamic analysis thread
"""
def run_dynamic_analysis(repo_name, strace_out_file, out_dir, rel_path, files, isPickle, isClean=False):
    # run strace
    for file in files:
        try:
            run_strace(repo_name, strace_out_file, out_dir, rel_path, file, isPickle, isClean)
        except Exception as e:
            print(f"Exception occurred for {repo_name}")
            traceback.print_exc()
            return False
    return True