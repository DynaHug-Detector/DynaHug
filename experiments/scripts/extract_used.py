import os
import tarfile
import pandas as pd
import ast

df = pd.read_csv("../malhug_result_info.csv", sep=",", encoding="ISO-8859-1")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Parse 'tags'
df["parsed_tags"] = df["tags"].apply(
    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
)

# Filter models with desired tag
target_tag = "region:us"
models_with_tag = df[df["parsed_tags"].apply(lambda tags: target_tag in tags)]
model_ids = models_with_tag["model_id/dataset_id"].tolist()
print(len(model_ids))

print(f"Found {len(model_ids)} models with tag '{target_tag}'.")

for model_id in model_ids:
    try:
        author, model_name = model_id.split("/")
        tarball_path = os.path.join(
            "/mnt/The_Second_Drive/Security/ML_Research/Malhug/malhug_result/model",
            author,
            model_name,
            f"{author}-{model_name}.tar.gz",
        )
        extract_path = os.path.join(
            "/mnt/The_Second_Drive/Security/ML_Research/Malhug/malhug_result/model",
            f"extracted_models_{target_tag}",
            author,
            model_name,
        )

        if os.path.exists(tarball_path):
            os.makedirs(extract_path, exist_ok=True)
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print(f"Extracted: {tarball_path} → {extract_path}")
        else:
            print(f"Missing file: {tarball_path}")
    except Exception as e:
        print(f"Error processing {model_id}: {e}")
