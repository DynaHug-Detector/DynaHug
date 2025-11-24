import subprocess
import os
import csv
import torch
from pickle import UnpicklingError

from fickling_safety import do_fickling_stuff
from model_tracer_runner import model_tracer_check


def modelscan_check(file_path, python_env):
    venv_path = python_env
    cmd = f"source {venv_path} && modelscan -p {file_path}"

    # Run the script using the other environment's Python
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    if "No issues found! 🎉" in result.stdout:
        return False
    else:
        return result.stdout


def picklescan_check(file_path):
    cmd = f"picklescan -p {file_path}"

    # Run the script using the other environment's Python
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    if "Infected files: 0" in result.stdout:
        return False
    else:
        return result.stdout


def do_open_source_checks(
    file_path, csv_file="opensource_tools_results.csv", python_env=None
):
    print("are we evne here")
    modelscan_result = modelscan_check(file_path, python_env)
    print(modelscan_result)
    if modelscan_result:
        print("modelscan thinks its the big bad, logging")

    fickling_severity, fickling_severity_cause = do_fickling_stuff(file_path)
    print("fickling_severity", fickling_severity)
    if (
        str(fickling_severity) == "Severity.OVERTLY_MALICIOUS"
        or str(fickling_severity) == "Severity.LIKELY_UNSAFE"
        or str(fickling_severity) == "Severity.LIKELY_OVERTLY_MALICIOUS"
        or str(fickling_severity) == "Severity.SUSPICIOUS"
        or str(fickling_severity) == "Severity.POSSIBLY_UNSAFE"
    ):
        print("fickling thinks its the big bad, loggging")

    picklescan_result = picklescan_check(file_path)
    print(picklescan_result)
    if picklescan_result:
        print("picklescan thinks its the big bad, logging")

    dynamic_result = model_tracer_check(file_path, "torch")
    print("modeltracer result", dynamic_result)

    # Prepare the CSV file
    file_exists = os.path.isfile(csv_file)

    # Define CSV headers
    headers = [
        "name",
        "picklescan_result",
        "modelscan_result",
        "fickling_result",
        "modeltracer_result",
        "fickling_category",
        "picklescan_output",
        "modelscan_output",
        "fickling_output",
    ]

    # Here "output" fields are based on your description
    row = {
        "name": file_path,
        "picklescan_result": bool(picklescan_result),
        "modelscan_result": bool(modelscan_result),
        "fickling_result": bool(fickling_severity),
        "modeltracer_result": dynamic_result,
        "fickling_category": str(fickling_severity),
        "picklescan_output": str(picklescan_result),
        "modelscan_output": str(modelscan_result),
        "fickling_output": str(fickling_severity_cause),
    }

    # Append to CSV
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    do_open_source_checks(
        "/mnt/The_Second_Drive/Security/ML_Research/pickleball/benign_test/pytorch_model_llm.bin"
    )
