import os
import pandas as pd
import pprint
import csv
import re
from collections import Counter
import argparse


def generalise_opcodes(base_dir, n_gram_size):
    # Dictionary to hold opcode counts per file
    file_opcodes = {}

    model_extensions = (".bin", ".pkl", ".pickle", ".pt", ".pth")

    def generalize_operand(operand_str):
        """
        Generalize operands by their type/pattern rather than specific values.
        """
        operand_str = operand_str.strip()

        if operand_str.isdigit() or (
            operand_str.startswith("-") and operand_str[1:].isdigit()
        ):
            return "NUMBER"
        elif operand_str.startswith("'") and operand_str.endswith("'"):
            return "STRING"
        elif operand_str.startswith('"') and operand_str.endswith('"'):
            return "STRING"
        elif operand_str in ["True", "False"]:
            return "BOOLEAN"
        elif operand_str == "None":
            return "NONE"
        elif re.match(r"^\d+\.\d+$", operand_str):
            return "FLOAT"
        elif operand_str.startswith("(") and operand_str.endswith(")"):
            return "TUPLE"
        elif operand_str.startswith("[") and operand_str.endswith("]"):
            return "LIST"
        elif operand_str.startswith("{") and operand_str.endswith("}"):
            return "DICT"
        elif "Storage" in operand_str:
            return "STORAGE_TYPE"
        elif operand_str == "MARK":
            return "MARK"
        elif operand_str.startswith("_var"):
            return "VAR"
        elif re.match(r"^\w+:", operand_str):
            return "DEVICE_STRING"
        elif operand_str.startswith("b'"):
            return "BINARY_DATA"
        else:
            words = operand_str.split()
            if words:
                return words[0].upper() if len(words[0]) > 2 else "OBJECT"
            return "OBJECT"

    def extract_generalized_opcodes(lines):
        """
        Extract opcodes and generalize their operands.
        """
        opcode_list = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line.isupper() and not line.startswith(
                "        "
            ):  # opcodes are not indented
                opcode = line

                # Look ahead to see if there are operand descriptions (indented lines)
                operand_descriptions = []
                j = i + 1

                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:  # empty line
                        j += 1
                        continue
                    elif (
                        next_line.startswith("Pushed ")
                        or next_line.startswith("Popped ")
                        or next_line.startswith("Memoized ")
                        or next_line.startswith("_var")
                    ):
                        operand_descriptions.append(next_line)
                        j += 1
                    # elif "import" in next_line:
                    #     operand_descriptions.append(next_line)
                    else:
                        # Next line is not an operand description, stop looking
                        break

                if operand_descriptions:
                    for desc in operand_descriptions:
                        if desc.startswith("Pushed "):
                            operand = desc[7:]  # Remove 'Pushed '
                            generalized_operand = generalize_operand(operand)
                            opcode_list.append(f"{opcode}_PUSHED_{generalized_operand}")

                        elif desc.startswith("Popped "):
                            operand = desc[7:]  # Remove 'Popped '
                            generalized_operand = generalize_operand(operand)
                            opcode_list.append(f"{opcode}_POPPED_{generalized_operand}")

                        elif desc.startswith("Memoized "):
                            # Extract the memoized pattern like "5064 -> '562'"
                            memo_match = re.match(r"Memoized (\d+) -> (.+)", desc)
                            if memo_match:
                                memo_value = memo_match.group(2)
                                generalized_value = generalize_operand(memo_value)
                                opcode_list.append(
                                    f"{opcode}_MEMOIZED_{generalized_value}"
                                )
                            else:
                                opcode_list.append(f"{opcode}_MEMOIZED_UNKNOWN")
                        elif desc.startswith("_var"):
                            opcode_list.append(f"{opcode}__VAR")
                        elif desc.contains("import"):
                            opcode_list.append(f"{opcode}_IMPORT")
                else:
                    # Opcode with no operand descriptions
                    opcode_list.append(opcode)

                i = j
            else:
                i += 1

        return opcode_list

    for path, folders, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".trace.txt"):
                file_path = os.path.join(path, filename)
                print(f"Processing: {file_path}")

                try:
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    opcode_list = extract_generalized_opcodes(lines)

                    # Generate n-grams from opcode list
                    opcodes = {}
                    for i in range(len(opcode_list) - n_gram_size + 1):
                        if n_gram_size == 1:
                            opcode = opcode_list[i]
                            key = "gen_" + str(opcode)
                            opcodes[key] = opcodes.get(key, 0) + 1
                        else:
                            # changing ngram to just normal string to eliminate possibility of bad typle usage
                            ngram_str = "_".join(opcode_list[i : i + n_gram_size])
                            key = f"gen_seq_{ngram_str}"
                            opcodes[key] = opcodes.get(key, 0) + 1
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

    # Output results
    pprint.pprint(file_opcodes)
    return file_opcodes


def write_to_csv(file_opcodes, n_gram_size, base_dir, TAG=None, malhug=None):
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
            print(filename)
            if "pytorch_model/" in filename:
                filename = "pytorch_model.bin"

        row = {"name": model_id, "filename": filename}

        for opcode in all_opcodes:
            row[opcode] = opcode_counts.get(opcode, 0)

        rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(
        os.path.join(base_dir, f"opcode_counts_wide_{n_gram_size}_generalised.csv"),
        index=False,
    )
    opcode_df = pd.read_csv(os.path.join(base_dir, "opcode_counts_wide_1.csv"))

    # Merge on the common key, e.g., 'file_path'
    combined_df = pd.merge(df, opcode_df, on="name", how="inner")
    combined_df.to_csv(os.path.join(base_dir, "merged_output.csv"), index=False)
    print(
        f"Wide format CSV saved to: {os.path.join(base_dir, f'opcode_counts_wide_{n_gram_size}_generalised.csv')}"
    )


def run_opcode_generalisor(base_dir, tag=None, malhug=None):
    for i in range(1, 4):
        opcodes = generalise_opcodes(base_dir, i)
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

    run_opcode_generalisor(args.base_dir, tag, args.malhug)
