import json
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Patch


def clean_tags(tags):
    """
    Clean and filter tags to remove invalid entries.
    Returns tuple of (cleaned_tags, tag_counts) where tag_counts is a list of numbers.
    """
    cleaned = []
    tag_counts = []

    for tag in tags:
        if (
            isinstance(tag, str)
            and tag.strip()
            and not tag.startswith("<function")
            and tag != "{}"
            and not tag.startswith("0x")
        ):
            tag_clean = tag.strip()

            match = re.search(r"^(.+?)\s+(\d+)$", tag_clean)
            if match:
                tag_name = match.group(1).strip()
                count = int(match.group(2))
                cleaned.append(tag_name)
                tag_counts.append(count)
            else:
                cleaned.append(tag_clean)
                tag_counts.append(0)  # Default to 0 if no number found

    return cleaned, tag_counts


def load_and_clean_data(filename):
    """
    Load JSON data and clean invalid entries.
    Returns tuple of (cleaned_data, all_tag_counts) where all_tag_counts contains the extracted numbers.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_data = {}
        all_tag_counts = {}

        for source, tags in data.items():
            if isinstance(tags, list):
                cleaned_tags, tag_counts = clean_tags(tags)
                if cleaned_tags:  # Only keep sources with valid tags
                    cleaned_data[source] = cleaned_tags
                    all_tag_counts[source] = tag_counts

        return cleaned_data, all_tag_counts

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return None, None


def analyze_tag_statistics(data, malhug_counts, tag_counts_data):
    """
    Perform comprehensive analysis of tag statistics.
    """
    if not data:
        print("No valid data to analyze.")
        return

    source_tags = {source: set(tags) for source, tags in data.items()}

    print(" MODEL CATEGORIES ANALYSIS")
    for source, counts in tag_counts_data.items():
        tags = data[source]
        print(f"\n  {source}:")
        print(f"   Tag counts list: {counts}")
        print(f"   Total extracted count: {sum(counts)}")

        for tag, count in zip(tags, counts):
            if count > 0:
                print(f"   • {tag}: {count}")

    print(f" Total sources analyzed: {len(source_tags)}")
    print(f" Sources: {', '.join(source_tags.keys())}\n")

    if len(source_tags) > 1:
        all_common_tags = set.intersection(*source_tags.values())
        print(" COMMON TAGS ACROSS ALL SOURCES")
        if all_common_tags:
            for tag in sorted(all_common_tags):
                count = malhug_counts.get(tag, 0)
                print(f"  • {tag} ({count} models)")
            print(f"\n Total common tags: {len(all_common_tags)}")
        else:
            print("   No tags are common across ALL sources")
    else:
        all_common_tags = set()
        print("  Only one source found - skipping common tags analysis")

    print(" PER-SOURCE STATISTICS")

    total_unique_tags = set()
    for source, tags in source_tags.items():
        total_unique_tags.update(tags)
        unique_to_source = tags - set().union(
            *(
                other_tags
                for other_source, other_tags in source_tags.items()
                if other_source != source
            )
        )

        print(f"\n  {source}:")
        print(f"   Total tags: {len(tags)}")
        print(f"   Unique to this source: {len(unique_to_source)}")
        if all_common_tags:
            print(f"   Tags in common with all: {len(tags & all_common_tags)}")

        if unique_to_source:
            sample_unique = sorted(list(unique_to_source))[:5]
            sample_with_counts = [
                f"{tag} ({malhug_counts.get(tag, 0)})" for tag in sample_unique
            ]
            print(f"   Sample unique tags: {', '.join(sample_with_counts)}")
            if len(unique_to_source) > 5:
                print(f"   ... and {len(unique_to_source) - 5} more")

    print(f"\n🌐 Total unique tags across all sources: {len(total_unique_tags)}")

    print(" TAG FREQUENCY ANALYSIS")

    tag_counts = Counter(tag for tags in source_tags.values() for tag in tags)

    print(f"\nMost Common Tags (appear in multiple sources):")
    common_tags = [(tag, count) for tag, count in tag_counts.most_common() if count > 1]

    if common_tags:
        for i, (tag, count) in enumerate(common_tags[:15]):  # Show top 15
            malhug_count = malhug_counts.get(tag, 0)
            print(f"  {i + 1:2d}. {tag:<35} ({count} sources, {malhug_count} models)")

        if len(common_tags) > 15:
            print(f"   ... and {len(common_tags) - 15} more common tags")
    else:
        print("No tags appear in multiple sources")

    print("CATEGORY GROUPING ANALYSIS")

    categories = analyze_tag_categories(total_unique_tags)

    for category, category_tags in categories.items():
        if len(category_tags) > 1:  # Only show categories with multiple tags
            print(f"\n{category} ({len(category_tags)} tags):")
            for tag in sorted(category_tags)[:10]:  # Show first 10
                sources_with_tag = [
                    source for source, tags in source_tags.items() if tag in tags
                ]
                malhug_count = malhug_counts.get(tag, 0)
                print(
                    f"   • {tag:<30} ({len(sources_with_tag)} sources, {malhug_count} models)"
                )
            if len(category_tags) > 10:
                print(f"   ... and {len(category_tags) - 10} more")


def analyze_tag_categories(tags):
    """
    Group tags into categories based on common patterns.
    """
    categories = {
        "Text/NLP": [],
        "Image/Vision": [],
        "Audio/Speech": [],
        "Video": [],
        "Generation": [],
        "Classification": [],
        "Detection": [],
        "Languages": [],
        "Other": [],
    }

    for tag in tags:
        tag_lower = tag.lower()

        languages = [
            "english",
            "chinese",
            "arabic",
            "turkish",
            "spanish",
            "french",
            "german",
            "hindi",
            "indonesian",
            "korean",
            "japanese",
            "portuguese",
            "russian",
            "bengali",
        ]
        if any(lang in tag_lower for lang in languages):
            categories["Languages"].append(tag)

        elif any(
            keyword in tag_lower
            for keyword in [
                "text",
                "nlp",
                "sentiment",
                "translation",
                "summarization",
                "named-entity",
                "word",
                "language",
                "conversational",
                "chat",
            ]
        ):
            categories["Text/NLP"].append(tag)

        elif any(
            keyword in tag_lower
            for keyword in [
                "image",
                "vision",
                "visual",
                "face",
                "object",
                "segmentation",
                "ocr",
            ]
        ):
            categories["Image/Vision"].append(tag)

        elif any(
            keyword in tag_lower for keyword in ["audio", "speech", "voice", "sound"]
        ):
            categories["Audio/Speech"].append(tag)

        elif "video" in tag_lower:
            categories["Video"].append(tag)

        elif any(
            keyword in tag_lower
            for keyword in ["generation", "generative", "synthesis", "to-"]
        ):
            categories["Generation"].append(tag)

        elif "classification" in tag_lower:
            categories["Classification"].append(tag)

        elif "detection" in tag_lower:
            categories["Detection"].append(tag)

        else:
            categories["Other"].append(tag)

    return categories


def save_analysis_report(
    data, malhug_counts, tag_counts_data, filename="analysis_report.txt"
):
    """
    Save the analysis results to a text file.
    """
    try:
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        analyze_tag_statistics(data, malhug_counts, tag_counts_data)

        report = captured_output.getvalue()
        sys.stdout = old_stdout

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        print(f" Analysis report saved to '{filename}'")

    except Exception as e:
        print(f" Error saving report: {e}")


def malhug_checker():
    """
    Load and process the malhug CSV to get tag counts.
    Returns a Counter object with tag counts.
    """
    try:
        df = pd.read_csv("../malhug_result_info.csv", sep=",", encoding="ISO-8859-1")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        print("Available columns:", df.columns.tolist())

        if "tags" in df.columns:
            df["parsed_tags"] = df["tags"].apply(
                lambda x: ast.literal_eval(x) if pd.notnull(x) else []
            )
            all_tags = [tag for sublist in df["parsed_tags"] for tag in sublist]
            tag_counts = Counter(all_tags)
            return tag_counts
        else:
            print("Column 'tags' not found.")
            return Counter()

    except FileNotFoundError:
        print("Error: '../malhug_result_info.csv' not found. Using empty counts.")
        return Counter()
    except Exception as e:
        print(f"Error processing malhug data: {e}. Using empty counts.")
        return Counter()


def main():
    """
    Main function to run the analysis.
    """
    filename = "model_categories.json"

    malhug_counts = malhug_checker()
    print(f" Loaded {len(malhug_counts)} unique tags from malhug data\n")

    data, tag_counts_data = load_and_clean_data(filename)

    if data and tag_counts_data:
        print(f" Successfully loaded data from {len(data)} sources\n")

        print("EXTRACTED TAG COUNTS SUMMARY:")
        for source, counts in tag_counts_data.items():
            non_zero_counts = [c for c in counts if c > 0]
            print(
                f"  {source}: {len(non_zero_counts)} tags with numbers, sum = {sum(counts)}"
            )
            print(f"    Full count list: {counts}")
        print()

        analyze_tag_statistics(data, malhug_counts, tag_counts_data)

        try:
            save_report = (
                input("\n Save analysis report to file? (y/n): ").lower().strip()
            )
            if save_report in ["y", "yes"]:
                save_analysis_report(data, malhug_counts, tag_counts_data)
        except KeyboardInterrupt:
            print("\n Analysis complete!")

    else:
        print(" Failed to load or clean data. Please check your JSON file.")


if __name__ == "__main__":
    main()
