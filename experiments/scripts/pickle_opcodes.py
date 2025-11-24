import pickletools


def print_all_pickle_opcodes():
    for opcode in pickletools.opcodes:
        print(f"{opcode.name:>10}: {opcode.code!r}")


print_all_pickle_opcodes()
with open("classifier/syscalls.txt", "w") as f:
    for opcode in pickletools.opcodes:
        f.write(f"{opcode.name}\n")
