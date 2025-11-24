import csv

csv_file_1 = "/placeholder"  # Replace with your first CSV file path
csv_file_2 = "/placeholder"  # Replace with your second CSV file path

# Output text file
output_file = "classifier/all_sequeunce_merged.txt"


def extract_headers(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


headers_1 = extract_headers(csv_file_1)
headers_2 = extract_headers(csv_file_2)

all_headers = sorted(set(headers_1) | set(headers_2))
all_headers.remove("filename")
all_headers.remove("name")
with open(output_file, "w") as f_out:
    for column in all_headers:
        f_out.write(column.strip() + "\n")

print(f"Extracted {len(all_headers)} unique column names to {output_file}")
