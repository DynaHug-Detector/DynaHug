import argparse
import opcode_combiner
import fickle_script
import opcode_analysis
import generalise_opcodes


def run_fickling(base_dir, filtering, filter_path=None):
    if filtering:
        if filter_path is None:
            print(
                "Please provide a csv to filter with, as the --filtering flag has been used"
            )
            exit()
        fickle_script.with_filtering(base_dir, filter_path)
    else:
        fickle_script.without_filtering(base_dir)


def run_combiner(base_dir, ngrams=None):
    if ngrams:
        opcode_combiner.n_gram_only(base_dir)
    else:
        opcode_combiner.everything(base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script to run everything in sequence"
    )
    parser.add_argument(
        "--base-dir", required=True, help="Point to directory of the model files"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="to generate the opcodes for the models in the giveni directory using fickling",
    )
    parser.add_argument("--filtering", help="to filter or not for fickling")
    parser.add_argument("--csv-path", help="Path to filtering csv for fickling")
    parser.add_argument(
        "--ngrams",
        action="store_true",
        help="to only do non-generalised features for static",
    )
    parser.add_argument(
        "--malhug",
        action="store_true",
        help="is the base directory malhug one or not",
    )
    args = parser.parse_args()
    if args.generate:
        run_fickling(args.base_dir, args.filtering, args.csv_path)
    opcode_analysis.run_opcode_generator(args.base_dir)
    generalise_opcodes.run_opcode_generalisor(args.base_dir)
    run_combiner(args.base_dir, args.ngrams)
