import pandas as pd

df1 = pd.read_csv("classifier/all_malhug_opcodes.csv", sep=",", encoding="ISO-8859-1")
df2 = pd.read_csv(
    "/mnt/The_Second_Drive/Security/ML_Research/pickleball/malicious/opcode_counts_wide_1.csv"
)

df1["combined"] = df1["name"].astype(str) + "/" + df1["filename"].astype(str)
df2["combined"] = df2["name"].astype(str) + "/" + df2["filename"].astype(str)

set1 = set(df1["combined"])
set2 = set(df2["combined"])

only_in_file1 = set1 - set2
only_in_file2 = set2 - set1

# Output results
print("In Malhug but not in Pickleball:")
for item in only_in_file1:
    print(item)

print("\nIn Pickleball but not in Malhug:")
for item in only_in_file2:
    print(item)
