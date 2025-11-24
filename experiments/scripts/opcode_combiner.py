import argparse
import pandas as pd
import os


def everything(base_dir):
    n_gram_size = 3
    base_dir = rf"{base_dir}"
    combined_dfs = []

    for i in range(1, n_gram_size + 1):
        print(f"Processing {i}-gram...")
        opcode_df = pd.read_csv(os.path.join(base_dir, f"opcode_counts_wide_{i}.csv"))
        generalised_df = pd.read_csv(
            os.path.join(base_dir, f"opcode_counts_wide_{i}_generalised.csv")
        )

        print(f"  Opcode {i}-gram shape: {opcode_df.shape}")
        print(f"  Generalized {i}-gram shape: {generalised_df.shape}")

        # Just merge the two types for this n-gram size
        combined_df = pd.merge(
            generalised_df,
            opcode_df,
            on=["name", "filename"],
            how="inner",
            suffixes=("_gen", "_reg"),
        )
        print(f"  Combined {i}-gram shape: {combined_df.shape}")
        combined_dfs.append(combined_df)

    print("\nChecking overlaps:")
    for i in range(len(combined_dfs) - 1):
        common = set(zip(combined_dfs[i]["name"], combined_dfs[i]["filename"])) & set(
            zip(combined_dfs[i + 1]["name"], combined_dfs[i + 1]["filename"])
        )
        print(f"Common files between {i + 1}-gram and {i + 2}-gram: {len(common)}")

    final_df = combined_dfs[0]
    print(f"Starting with {final_df.shape}")

    for i, df in enumerate(combined_dfs[1:], 2):
        print(f"Before merging {i}-gram: {final_df.shape}")
        final_df = pd.merge(
            final_df,
            df,
            on=["name", "filename"],
            how="inner",
        )
        print(f"After merging {i}-gram: {final_df.shape}")

    output_path = os.path.join(base_dir, f"combined_ngrams_{n_gram_size}.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")
    print(f"Final shape: {final_df.shape}")

    return final_df


def n_gram_only(base_dir):
    n_gram_size = 3
    base_dir = rf"{base_dir}"
    combined_dfs = []
    for i in range(1, n_gram_size):
        print(i)
        opcode_df = pd.read_csv(os.path.join(base_dir, f"opcode_counts_wide_{i}.csv"))
        generalised_df = pd.read_csv(
            os.path.join(base_dir, f"opcode_counts_wide_{i + 1}.csv")
        )

        common_cols = list(set(opcode_df.columns) & set(generalised_df.columns))

        combined_df = pd.merge(
            generalised_df,
            opcode_df,
            on=common_cols,
            how="inner",
        )
        combined_dfs.append(combined_df)

    final_df = combined_dfs[0]
    for df in combined_dfs[1:]:
        common_cols = list(set(final_df.columns) & set(df.columns))
        final_df = pd.merge(final_df, df, on=common_cols, how="inner")

    final_df.to_csv(
        os.path.join(base_dir, f"opcode_counts_combined_{n_gram_size}.csv"), index=False
    )
    print(
        f"Wide format CSV saved to: {os.path.join(base_dir, f'opcode_counts_combined_{n_gram_size}.csv')}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="combine opcodes depending on which combo you want"
    )
    parser.add_argument(
        "--ngram",
        action="store_true",
        help="Category of the opcodes being combined, use if you only want to use non-generalised features",
    )
    parser.add_argument("--base-dir", help="directory to extract opcodes in")
    args = parser.parse_args()
    if args.ngram:
        n_gram_only(args.base_dir)

    else:
        everything(args.base_dir)
