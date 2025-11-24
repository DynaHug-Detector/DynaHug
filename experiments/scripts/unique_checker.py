import pandas as pd
import ast
import json

# === Step 1: Load and clean the CSV ===
df = pd.read_csv("../malhug_result_info.csv", sep=",", encoding="ISO-8859-1")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Available columns:", df.columns.tolist())

# === Step 2: Parse the 'tags' column ===
if "tags" in df.columns:
    df["parsed_tags"] = df["tags"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
else:
    print("Column 'tags' not found.")
    exit()

# === Step 3: Load category definitions from JSON ===
with open("../huggingface_model_categories.json", "r", encoding="utf-8") as f:
    category_data = json.load(f)

# === Step 4: Check if any item has multiple tags in the same category ===
for category, tag_dict in category_data.items():
    print(f"\nChecking category: {category}")

    # Set of all tags in this category
    category_tags = set(tag_dict.keys())

    # For each row, filter tags to those belonging to this category
    df[category + "_tags"] = df["parsed_tags"].apply(
        lambda tags: [t for t in tags if t in category_tags]
    )

    # Check if any row has multiple tags in this category
    rows_with_multiple_tags = df[df[category + "_tags"].apply(len) > 1]

    if rows_with_multiple_tags.empty:
        print(
            f"Category '{category}' is unique: no item has multiple tags in this category."
        )
    else:
        print(f"Category '{category}' is NOT unique: some items have multiple tags.")
        print("Examples of items with multiple tags in this category:")
        # Show up to 5 example rows with multiple tags
        print(rows_with_multiple_tags[[category + "_tags"]].head())

# Clean up (optional)
for category in category_data.keys():
    if category + "_tags" in df.columns:
        df.drop(columns=[category + "_tags"], inplace=True)
#
# import json
# import json
# import pandas as pd
from huggingface_hub import HfApi

# Load category definitions
with open("../huggingface_model_categories.json", "r", encoding="utf-8") as f:
    category_data = json.load(f)

api = HfApi()


def fetch_top_models(n=2000, sort="likes"):
    models = []
    page_size = 100
    for i in range((n + page_size - 1) // page_size):
        print(f"Fetching page {i + 1}")
        page_models = api.list_models(sort=sort, limit=page_size, full=True)
        models.extend(page_models)
        if len(models) >= n:
            break
    return models[:n]


print("Fetching top 2000 models from Hugging Face...")
models = fetch_top_models(5000)
print(f"Fetched {len(models)} models")

data = []
for m in models:
    tags = m.tags if m.tags else []
    data.append({"model_id": m.modelId, "tags": tags})

df = pd.DataFrame(data)

# For each category, check how many models have multiple tags,
# and how many models have any tags in that category
for category, tag_dict in category_data.items():
    print(f"\nChecking category: {category}")

    category_tags = set(tag_dict.keys())

    # Extract tags for this category
    df[category + "_tags"] = df["tags"].apply(
        lambda tags: [t for t in tags if t in category_tags]
    )

    # Models with at least one tag in this category
    models_with_tags = df[df[category + "_tags"].apply(len) > 0]

    # Models with multiple tags in this category
    models_with_multiple = models_with_tags[
        models_with_tags[category + "_tags"].apply(len) > 1
    ]

    total_models_with_tags = len(models_with_tags)
    total_models_with_multiple = len(models_with_multiple)

    if total_models_with_tags == 0:
        print(f"No models have tags in category '{category}'.")
    else:
        print(f"Models with tags in '{category}': {total_models_with_tags}")
        print(
            f"Models with multiple tags in '{category}': {total_models_with_multiple}"
        )
        ratio = total_models_with_multiple / total_models_with_tags
        print(f"Ratio of models with multiple tags: {ratio:.2%}")

        if total_models_with_multiple > 0:
            print("Examples of models with multiple tags:")
            print(models_with_multiple[["model_id", category + "_tags"]].head())

# Cleanup columns
for category in category_data.keys():
    df.drop(columns=[category + "_tags"], inplace=True)

# Check for uniqueness in pytorch + pipeline_tag combination
pipeline_tags = set(category_data.get("pipeline_tag", {}).keys())

# Add pipeline_tag filtering column
df["pipeline_tag_tags"] = df["tags"].apply(
    lambda tags: [t for t in tags if t in pipeline_tags]
)

# Filter models with pytorch
pytorch_models = df[df["tags"].apply(lambda tags: "pytorch" in tags)]

# Filter models that have more than one pipeline_tag
non_unique_pipeline_tag_models = pytorch_models[
    pytorch_models["pipeline_tag_tags"].apply(len) > 1
]

total_pytorch_models = len(pytorch_models)
total_non_unique = len(non_unique_pipeline_tag_models)

print("\n=== Pytorch + Pipeline Tag Uniqueness Check ===")
print(f"Total models with pytorch tag: {total_pytorch_models}")
print(f"Models with more than one pipeline_tag: {total_non_unique}")

if total_pytorch_models > 0:
    ratio_non_unique = total_non_unique / total_pytorch_models
    print(f"Percentage of non-unique models: {ratio_non_unique:.2%}")

if total_non_unique > 0:
    print("\nExamples of non-unique models (pytorch + multiple pipeline_tags):")
    print(non_unique_pipeline_tag_models[["model_id", "pipeline_tag_tags"]].head())

# Drop helper column
df.drop(columns=["pipeline_tag_tags"], inplace=True)

