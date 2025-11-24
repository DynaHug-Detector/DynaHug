import re
from pathlib import Path

import random


def generate_paylaod(file_path):
    with open(file_path, "rb") as f:
        payload = f.read()

    pattern = re.compile(b"q.", re.DOTALL)

    print(payload)
    pattern = re.compile(b"q(.)", re.DOTALL)

    matches = pattern.findall(payload)

    print("Bytes after 'q':")
    for m in matches:
        print(m, "->", int.from_bytes(m, "big"))
    hex_values = [b.hex() for b in matches]
    print(hex_values)  # ['01', 'ff', '41']

    def replacer(match):
        byte_after_q = match.group(1)  # captured byte
        return b"r" + byte_after_q + b"\x00\x10\x11"

    # Perform replacement
    new_payload = pattern.sub(replacer, payload)

    print(new_payload)
    print(new_payload[2:-1])
    new_payload = b"(" + new_payload[2:-1] + b"1"
    print(new_payload)
    return new_payload


def get_random_pkl_file(folder_path, exclude_dirs=["failed"], recursive=False):
    folder = Path(folder_path)
    pattern = "**/*.pkl" if recursive else "*.pkl"
    pkl_files = list(folder.glob(pattern))

    if exclude_dirs:
        exclude_dirs = [Path(d).resolve() for d in exclude_dirs]
        pkl_files = [
            f
            for f in pkl_files
            if not any(ex_dir in f.resolve().parents for ex_dir in exclude_dirs)
        ]
    if not pkl_files:
        return None

    return random.choice(pkl_files)


if __name__ == "__main__":
    generate_paylaod(get_random_pkl_file("payloads/"))
