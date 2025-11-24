import csv
import os
import pandas as pd
import sys


csv.field_size_limit(sys.maxsize)


def print_scan_results(csv_file_path):
    name_count = 0
    picklescan_result = 0
    modelscan_result = 0
    fickling_result = 0
    modeltracer_result = 0
    sneaky_model = []

    seen_names = set()  # Track already processed names

    df = pd.read_csv(
        "/home/zol/School/Research/scripts/classifier/text_gen_injected_pypi_opcodes.csv"
    )

    # Normalize and prepare model names from reference CSV
    models_to_scan = (
        df["name"].str.strip().str.replace("/", "__") + "/" + df["filename"].str.strip()
    ).tolist()

    print("Models to scan:", models_to_scan)

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            raw_name = row.get("name", "").strip()

            if "downloaded_injected" in raw_name:
                parts = raw_name.split(os.sep)
                try:
                    idx = parts.index("downloaded_injected")
                    name = f"{parts[idx + 1]}/{parts[idx + 2]}"
                except IndexError:
                    name = raw_name  # fallback
            else:
                name = raw_name

            if name not in models_to_scan:
                continue  # Skip irrelevant models

            if name in seen_names:
                continue  # Already analyzed

            seen_names.add(name)  # Mark as seen

            print(f"Analyzing model: {name}")
            name_count += 1

            modelscan = row.get("modelscan_result", "N/A")
            if modelscan == "True":
                modelscan_result += 1
            else:
                sneaky_model.append(name)

            picklescan = row.get("picklescan_result", "N/A")
            if picklescan == "True":
                picklescan_result += 1

            fickling = row.get("fickling_result", "N/A")
            if fickling == "True":
                fickling_result += 1

            modeltracer = row.get("modeltracer_result", "N/A")
            if modeltracer == "True":
                modeltracer_result += 1

            print(f"\nName: {name}")
            print(f"  ModelScan Result:     {modelscan}")
            print(f"  PickleScan Result:    {picklescan}")
            print(f"  Fickling Result:      {fickling}")
            print(f"  ModelTracer Result:   {modeltracer}")

    # Final summary
    print(f"\nTotal unique models analyzed: {name_count}")
    print(f"ModelScan found malicious: {modelscan_result}")
    print(f"Picklescan found malicious: {picklescan_result}")
    print(f"Fickling found malicious: {fickling_result}")
    print(f"Modeltracer found malicious: {modeltracer_result}")
    print("Sneaky models:", sneaky_model)
    print("missing moels", set(models_to_scan) - set(seen_names))


def print_all_scan_results(csv_file_path):
    name_count = 0
    picklescan_result = 0
    modelscan_result = 0
    fickling_result = 0
    modeltracer_result = 0
    sneaky_model = []

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            name = row.get("name", "").strip()
            if name:  # count only non-empty names
                name_count += 1

                modelscan = row.get("modelscan_result", "N/A")
                if modelscan == "True":
                    modelscan_result += 1
                else:
                    sneaky_model.append(name)
                picklescan = row.get("picklescan_result", "N/A")
                if picklescan == "True":
                    picklescan_result += 1
                fickling = row.get("fickling_result", "N/A")
                if fickling == "True":
                    fickling_result += 1
                modeltracer = row.get("modeltracer_result", "N/A")
                if modeltracer == "True":
                    modeltracer_result += 1

                print(f"\nName: {name}")
                print(f"  ModelScan Result:     {modelscan}")
                print(f"  PickleScan Result:    {picklescan}")
                print(f"  Fickling Result:      {fickling}")
                print(f"  ModelTracer Result:   {modeltracer}")

    print(f"\nTotal items in 'name' column: {name_count}")  # Example usage:
    print(f"\nModelScan found malicious: {modelscan_result}")
    print(f"\nPicklescan found malicious: {picklescan_result}")
    print(f"\nFickling found malicious: {fickling_result}")
    print(f"\nModeltracer found malicious: {modeltracer_result}")
    print("sneaky model:", sneaky_model)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python print_scan_results.py <path_to_csv>")
    else:
        csv_path = sys.argv[1]
        print_all_scan_results(csv_path)
