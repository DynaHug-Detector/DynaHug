import argparse
import os
import pprint
import pandas as pd
import csv
from collections import Counter


def generate_opcodes(base_dir, n_gram_size):
    file_opcodes = {}
    # Valid model file extensions
    model_extensions = (".bin", ".pkl", ".pickle", ".pt", ".pth")

    for path, folders, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".trace.txt"):
                file_path = os.path.join(path, filename)
                print(f"Processing: {file_path}")

                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    # Extract opcode lines (lines starting with uppercase letter)
                    opcode_list = [
                        line.strip() for line in lines if line and line[0].isupper()
                    ]
                    # opcodes = {}
                    # for line in lines:
                    #     if line and line[0].isupper():
                    #         opcodes[line] = opcodes.get(line, 0) + 1
                    #
                    # Generate n-grams from opcode list
                    opcodes = {}
                    for i in range(len(opcode_list) - n_gram_size + 1):
                        if n_gram_size == 1:
                            opcode = opcode_list[i]
                            opcodes[opcode] = opcodes.get(opcode, 0) + 1
                        else:
                            # changing ngram to just normal string to eliminate possibility of bad typle usage
                            ngram_str = "_".join(opcode_list[i : i + n_gram_size])
                            key = f"seq_{ngram_str}"
                            opcodes[key] = opcodes.get(key, 0) + 1  # print(opcodes)
                    model_size = None
                    found_model = None

                    base_model_name = filename.rsplit(".trace.txt", 1)[0]
                    exact_model_path = os.path.join(path, base_model_name)

                    if (
                        os.path.isfile(exact_model_path)
                        and os.path.basename(exact_model_path) != "pytorch_model.bin"
                    ):
                        found_model = exact_model_path
                        model_size = os.path.getsize(found_model)

                    if found_model is None:
                        for fname in os.listdir(path):
                            if fname != "pytorch_model.bin" and any(
                                fname.endswith(ext) for ext in model_extensions
                            ):
                                fallback_model_path = os.path.join(path, fname)
                                found_model = fallback_model_path
                                model_size = os.path.getsize(found_model)
                                break

                    if found_model is None:
                        pt_model_path = os.path.join(path, "pytorch_model.bin")
                        if os.path.isfile(pt_model_path):
                            found_model = pt_model_path
                            model_size = os.path.getsize(found_model)

                    file_opcodes[file_path] = {
                        "opcodes": opcodes,
                    }

                except Exception as e:
                    print(f"Failed to process '{file_path}': {type(e).__name__}: {e}")
                    print()  # Output results
    pprint.pprint(file_opcodes)
    return file_opcodes


def write_to_csv(file_opcodes, n_gram_size, base_dir, TAG=None, malhug=None):
    #
    # output_file = os.path.join(base_dir, "opcode_counts.txt")
    # with open(output_file, "w") as f:
    #     pprint.pprint(file_opcodes, stream=f)
    #
    # print(f"\nOpcode counts written to: {output_file}")
    # Output to CSV

    all_opcodes = set()
    for entry in file_opcodes.values():
        if n_gram_size == 1:
            all_opcodes.update(k.strip() for k in entry["opcodes"].keys())
        else:
            all_opcodes.update(k for k in entry["opcodes"].keys())

    rows = []
    for file_path, entry in file_opcodes.items():
        if n_gram_size == 1:
            opcode_counts = {k.strip(): v for k, v in entry["opcodes"].items()}
        else:
            opcode_counts = {k: v for k, v in entry["opcodes"].items()}

        # Extract relative parts of the path
        if malhug:
            try:
                rel_path = file_path.split(f"{TAG}/")[1]
                parts = rel_path.split(os.sep)
                print(parts)
                model_id = "/".join(
                    parts[:2]
                )  # e.g., "Madans/twitter-roberta-base-sentiment-unsafe"
                # if parts[-1]
                filename = "/".join(parts[3:])  # e.g., "unsafe_model.pt.trace.txt"
                filename = filename.split(".trace.txt")[0]
            except IndexError:
                model_id = "UNKNOWN"
                filename = os.path.basename(file_path)
        else:
            # change this to right before the model name starts, so a path like "/mnt/downloaded_injected/star23__newstart/" would be "injected/"
            rel_path = file_path.split(f"{base_dir}")[1]
            parts = rel_path.split(os.sep)
            print(parts)
            model_id = parts[0].replace("__", "/")
            filename = "/".join(parts[1:])
            filename = filename.split(".trace.txt")[0]
            if "pytorch_model/" in filename:
                filename = "pytorch_model.bin"

        # Build row
        row = {"name": model_id, "filename": filename}

        # Add opcode counts
        for opcode in all_opcodes:
            row[opcode] = opcode_counts.get(opcode, 0)

        rows.append(row)

    # __import__("pprint").pprint(rows)
    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(base_dir, f"opcode_counts_wide_{n_gram_size}.csv"), index=False
    )

    print(
        f"\nWide-format opcode counts written to: opcode_counts_wide_{n_gram_size}.csv"
    )


def run_opcode_generator(base_dir, tag=None, malhug=None):
    for i in range(1, 4):
        opcodes = generate_opcodes(base_dir, i)
        write_to_csv(opcodes, i, base_dir, tag, malhug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse opcodes from files")

    parser.add_argument(
        "--malhug",
        action="store_true",
        help="is current directory the malhug one or not",
    )
    parser.add_argument("--base-dir", help="directory to analyse opcodes in")

    parser.add_argument(
        "--TAG", help="only useful for malhug, indicate tag of malhug dataset"
    )
    args = parser.parse_args()

    if args.base_dir is None:
        print("please provide base-dir")
        exit()
    tag = args.TAG

    run_opcode_generator(args.base_dir, tag, args.malhug)
