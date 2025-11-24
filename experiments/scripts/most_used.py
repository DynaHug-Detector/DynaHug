import pandas as pd
import ast
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Patch

df = pd.read_csv("../malhug_result_info.csv", sep=",", encoding="ISO-8859-1")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Available columns:", df.columns.tolist())

if "tags" in df.columns:
    df["parsed_tags"] = df["tags"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    all_tags = [tag for sublist in df["parsed_tags"] for tag in sublist]
    tag_counts = Counter(all_tags)
else:
    print("Column 'tags' not found.")
    exit()

with open("../huggingface_model_categories.json", "r", encoding="utf-8") as f:
    category_data = json.load(f)

category_plot_data = []

for category, items in category_data.items():
    for tag, label in items.items():
        count = tag_counts.get(tag, 0)
        if count > 0:
            category_plot_data.append(
                {"Category": category, "Tag": tag, "Label": label, "Count": count}
            )

plot_df = pd.DataFrame(category_plot_data)


plot_df = plot_df.sort_values(by=["Category", "Count"], ascending=[True, False])

y_labels = []
y_positions = []
bar_colors = []
bar_heights = []
legend_handles = {}

y = 0  # y-axis counter

cmap = cm.get_cmap("tab20")
category_colors = {}

for idx, category in enumerate(plot_df["Category"].unique()):
    sub_df = plot_df[plot_df["Category"] == category]

    # Assign a color to this category
    color = cmap(idx % 20)
    category_colors[category] = color

    for _, row in sub_df.iterrows():
        y_labels.append(f"{row['Tag']}")
        y_positions.append(y)
        bar_colors.append(color)
        bar_heights.append(row["Count"])
        y += 1

    # Add gap after each category
    y += 1  # leave one space

# Plot
plt.figure(figsize=(14, 10))
bars = plt.barh(y_positions, bar_heights, color=bar_colors)

plt.yticks(ticks=y_positions, labels=y_labels)
plt.xlabel("Number of Models")
plt.title("Model Counts Grouped by Category and Tag")

# Build legend

legend_patches = [
    Patch(color=color, label=cat) for cat, color in category_colors.items()
]
plt.legend(handles=legend_patches, title="Category", loc="best")

plt.tight_layout()
plt.savefig("../models_grouped.png")
# plt.show()
