import argparse
import os
import shutil
import sys
import pandas as pd
from utils.utils import (
    parse_files,
    find_files_with_repo_names,
    create_analysis_archive,
    remove_file_from_directory,
    upload_to_gcs,
    get_model_metadata,
    list_files_gcs,
    print_system_info,
    get_model_metadata_from_api,
    download_from_gcs,
    validate_folders,
    extract_analysis_archive,
    get_folders_from_directory
)
from src.metadata_collector import MetadataCollector
from src.analysis import run_dynamic_analysis, OUTPUT_DIR, CLEAN_DATASET, MALICIOUS_DATASET
from src.strace_analyzer import StraceAnalyzer
from classifier.svm import SyscallAnomalyDetector
from src.download import download_models_with_space_limit, SUPPORTED_FORMATS, download_from_old_list
import time
import atexit
import traceback
import gc
import ast
import random
from itertools import chain
import json

# Constants
CSV_FILE = "data/malhug_result_info.csv"
LIMIT = 10000 # Limit for the number of models to analyze
OFFSET = 0 # To help in executing models in batches in malicious samples
DOWNLOAD_COMPONENT = "download"
UPLOAD_COMPONENT = "gcsupload"
CLASSIFICATION_COMPONENT = "classifier"
REMOVE_EXPLORED = 'remove-explored' # Removes already explored models from being uploaded again
DYNAMIC_ANALYSIS_COMPONENT = "dynamic-analysis"
CLEANUP_COMPONENT = "cleanup"
PIPELINE = [DOWNLOAD_COMPONENT, DYNAMIC_ANALYSIS_COMPONENT, CLASSIFICATION_COMPONENT, REMOVE_EXPLORED, UPLOAD_COMPONENT, CLEANUP_COMPONENT]

def run_malicious_samples(isPickle, log_dir, out_dir, wild_run=False, injected_models=False):
    """
    Run dynamic analysis for malicious samples
    """
    count = {"Success": [], "Failure": []} # Count of how many pickle files failed or succeeded to go through the dynamic analysis process
    # Iterate through csv file and filter for pickle files
    if wild_run or injected_models:
        suffix = TAG if wild_run else f"injected_models/{TAG}"
        rows = list(
                chain.from_iterable([find_files_with_repo_names(f"*{ext}", f"{MALICIOUS_DATASET}/model/{suffix}") for ext in SUPPORTED_FORMATS]
            )) if isPickle else find_files_with_repo_names("*.py", f"{MALICIOUS_DATASET}/dataset/{suffix}")
        
    else:  
        # MALHUG analysis
        df = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")
        if isPickle:
            filtered_df = df[df["model_type"].fillna("").str.contains("Pickle|PyTorch", regex=True)] 
        else:
            # Choosing only dataset models
            filtered_df = df[df["type"] == "dataset"]

        # Keep models with the specified TAG
        if TAG != "all":
            parsed_tags = df["tags"].apply(
                lambda x: ast.literal_eval(x) if pd.notnull(x) else []
            )

            # Filter models with desired tag
            filtered_df = filtered_df[parsed_tags.apply(lambda tags: TAG in tags)]

        rows = []
        for i, row in filtered_df.iterrows():
            model_id = row["model_id/dataset_id"]
            files = parse_files(row["files"])
            
            # Removing duplicates
            mal_files = set()
            for file in files:
                _, ext = os.path.splitext(file)
                if ext in SUPPORTED_FORMATS:
                    fname = os.path.basename(file)
                    if fname not in mal_files:
                        mal_files.add(fname)
            
            rows += list(
                chain.from_iterable([find_files_with_repo_names(f, f"{MALICIOUS_DATASET}/model/MALHUGxPickleball/{model_id}", model_id) for f in mal_files]
            ))

    i = OFFSET
    num_rows = len(rows)

    while i < min(LIMIT + OFFSET, num_rows):
        repo_name, relative_path, file = rows[i]
        analysis_type = "model" if isPickle else "dataset"
        log_malicious_dir = os.path.join(log_dir, "malicious-straces", analysis_type, TAG, "wild" if wild_run else "", "injected" if injected_models else "")
        os.makedirs(log_malicious_dir, exist_ok=True)
        rel_path = relative_path.replace("/", "__")
        if run_dynamic_analysis(repo_name, os.path.join(log_malicious_dir, f"strace_logs_{repo_name.replace('/', '__')}--{rel_path}"), out_dir, rel_path, [file], isPickle, isClean=True):
            count["Success"].append(f"{repo_name}_{rel_path}")
        else:
            count["Failure"].append(f"{repo_name}_{rel_path}")
        i += 1

        print(f"{count['Success']} {analysis_type} files have successfully gone through the dynamic analysis process and {count['Failure']} have failed.")

def run_clean_samples(isPickle, log_dir, out_dir, wild_run=False):
    """
    Run dynamic analysis for benign samples
    """
    count = {"Success": [], "Failure": []} # Count of how many pickle files failed or succeeded to go through the dynamic analysis process
    # Iterate through csv file and filter for pickle files
    if isPickle:
        rows = list(
                chain.from_iterable([find_files_with_repo_names(f"*{ext}", f"{CLEAN_DATASET}/model/{TAG}") for ext in SUPPORTED_FORMATS]
            ))
    else:
        rows = find_files_with_repo_names("*.py", f"{CLEAN_DATASET}/dataset/{TAG}")
    num_rows = len(rows)
    i = OFFSET

    while i < min(LIMIT + OFFSET, num_rows):
        repo_name, relative_path, file = rows[i]
        analysis_type = "model" if isPickle else "dataset"
        log_clean_dir = os.path.join(log_dir, "clean-straces", analysis_type, TAG, "wild" if wild_run else "normal")
        os.makedirs(log_clean_dir, exist_ok=True)
        rel_path = relative_path.replace("/", "__")
        if run_dynamic_analysis(repo_name, os.path.join(log_clean_dir, f"strace_logs_{repo_name.replace('/', '__')}--{rel_path}"), out_dir, rel_path, [file], isPickle=isPickle, isClean=True):
            count["Success"].append(f"{repo_name}_{rel_path}")
        else:
            count["Failure"].append(f"{repo_name}_{rel_path}")
        i += 1

        print(f"{count['Success']} {analysis_type} files have successfully gone through the dynamic analysis process and {count['Failure']} have failed.")

def exit_handler(metadata_collector):
    """
    General purpose cleanup function
    """
    metadata_collector.save()
    print("CSV has been saved.")

def main():
    parser = argparse.ArgumentParser(description="Download models with space constraints, run analysis, and upload to GCS")
    parser.add_argument("--tag", required=True, help="Tag to filter models")
    parser.add_argument("--num-models", type=int, default=LIMIT, help="Number of models to download and analyze")
    parser.add_argument("--space-limit-gb", type=float, default=100, help="Space limit in GB")
    parser.add_argument("--bucket-name", help="GCS bucket name")
    parser.add_argument("--work-dir", default="./", help="Working directory for downloads and logs")
    parser.add_argument("--file-type", choices=["benign", "malicious", "both"], default="benign", help="Type of file to analyze (benign or malicious or both)")
    parser.add_argument("--isModel", action="store_true", help="Run analysis on models or datasets")
    parser.add_argument('--run-inference', action="store_true", help='Deserialize pickle files or run datasets')
    parser.add_argument('--save-count-per-sample', action='store_true',  help='Save the count of system calls per sample to CSV files')
    parser.add_argument('--wild-run', action='store_true', help='Run the already trained classifier on wild models from Hugging Face')
    parser.add_argument("--feature-types", nargs='+', default=["presence", "frequency"], help="Types of features to extract from syscall data (presence, frequency, sequence)")
    parser.add_argument("--retrieve-mal-models", action="store_true", help="Retrieve malicious models from Hugging Face using the detection from hugging face security tools")
    parser.add_argument("--active-components", nargs='+', default=PIPELINE, help=f"Active components in the pipeline to run ({','.join(PIPELINE)})")
    parser.add_argument("--injected-models", action="store_true", help="Process injected models stored in the GCP bucket")
    parser.add_argument("--checksum-verify", help="Verifying the checksums of a given checklist of models", action="store_true")
    parser.add_argument("--mode", required=True, choices=['pytorch', 'generalized', 'pytorch_model.bin'],help="Mode of filtering models during download. Can be 'pytorch', 'pytorch_model.bin' or 'generalized'. 'pytorch' mode extracts model files which hugging face would extract with their from_pretrained function. 'generalized' function would extract all valid pickle files with the SUPPORTED_FORMATS extension. 'pytorch_model.bin' as the name suggests, only picks up pytorch_model.bin files from the repository.")
    parser.add_argument("--download_from_csv", action='store_true', help="Mode for using the previous set of model names for collecting new traces afresh")
    parser.add_argument("--best_model_path", help="The best model path for the classification component in the pipeline.")

    args = parser.parse_args()
    
    global TAG
    TAG = args.tag
    BEST_MODEL_PATH = "classifier/models/text-generation/1000_benign_data_presence_frequency_sequence_presence_sequence_frequency_best/IsolationForest" # Change this to the actual path
    clf_name = os.path.join(os.path.basename(os.path.dirname(BEST_MODEL_PATH)), os.path.basename(BEST_MODEL_PATH))
    work_dir = os.path.abspath(args.work_dir)
    download_dir = os.path.join(work_dir, "data", "clean_dataset" if args.file_type == "benign" else "malicious_dataset", "model" if args.isModel else "dataset", "injected_models" if args.injected_models else "",TAG)
    log_dir = os.path.join(work_dir, "logs")
    download_log = os.path.join(args.work_dir, "data", "clean_dataset" if args.file_type == "benign" else "malicious_dataset", "model" if args.isModel else "dataset", f"{TAG}_downloaded_models.csv")
    out_dir = os.path.join(work_dir, OUTPUT_DIR, "clean-output" if args.file_type == "benign" else "malicious-output", "model" if args.isModel else "dataset", TAG, "wild" if args.wild_run else "", "injected" if args.injected_models else "")
    metadata_dir = os.path.join(work_dir, "metadata", "model" if args.isModel else "dataset", TAG, clf_name)
    time_taken_log = os.path.join(metadata_dir, f"time_taken.csv")
    anomaly_log = os.path.join(metadata_dir, f"anomalous_models.csv")
    wild_mal_log = os.path.join(os.path.dirname(download_log), "wild_malicious_models_api.csv")
    wild_download_log = os.path.join(args.work_dir, "data", "clean_dataset" if args.file_type == "benign" else "malicious_dataset", "model" if args.isModel else "dataset", f"wild_{TAG}_downloaded_models.csv")
    injected_model_download_log = os.path.join(args.work_dir, "data", "malicious_dataset", "model" if args.isModel else "dataset", "injected_models", f"injected_{TAG}_downloaded_models.csv")
    old_download_log = os.path.join(args.work_dir, "data/clean_dataset/model/text-generation_downloaded_models_copy.csv")

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    space_limit_bytes = int(args.space_limit_gb * 1024 * 1024 * 1024)
    
    output_path = os.path.join(OUTPUT_DIR, "dynamic_analysis_print_statements.log")

    old_download_log_df = get_model_metadata(old_download_log, ["name", "filename", "likes", "downloads"])
    download_log_df = get_model_metadata(download_log, ["name", "filename", "likes", "downloads", "last_updated","protectAiScan", "avScan", "pickleImportScan", "jFrogScan"]) # Download log for training
    wild_download_log_df = get_model_metadata(wild_download_log, ["name", "filename", "likes", "downloads"]) # download log for wild run
    injected_model_download_log_df = get_model_metadata(injected_model_download_log, ["name", "filename", "likes", "downloads", "last_updated"]) # download log for injected models
    download_df = None
    if args.wild_run:
        successful_downloaded_models = wild_download_log_df["name"].tolist()
        download_df = wild_download_log_df
    elif args.injected_models:
        successful_downloaded_models = injected_model_download_log_df["name"].tolist()
        download_df = injected_model_download_log_df
    else:
        download_df = download_log_df
        successful_downloaded_models = download_log_df["name"].tolist()
    
    explored_models = set(successful_downloaded_models)
    last_success_download = successful_downloaded_models[-1] if successful_downloaded_models else None
    # last_success_download = "Ranger/Dial0GPT-small-harrypotter" # For testing purposes
    anomalous_df = get_model_metadata(anomaly_log, ["name", "filename", "likes", "downloads", "last_updated", "decision_score"]) # Set of models detected as anomaly
    
    classified_set = set() # Set of model names which have already been classified.
    uploaded_set = set() # Set of model names which have already been uploaded to GCS
    
    # Check if the state when the program state needs to restored or not
    if successful_downloaded_models:
        print(f"Restoring state from last successful download: {last_success_download}")
        classified_set.update(successful_downloaded_models)
        uploaded_set.update(successful_downloaded_models)
    else:
        print("No previous state found. Starting fresh run.")

    metadata_collector = MetadataCollector(
        time_log_path=time_taken_log,
        anomaly_log_path=anomaly_log,
        wild_mal_log_path=wild_mal_log if args.retrieve_mal_models else None,
        wild_run=args.wild_run,
        retrieve_mal_models=args.retrieve_mal_models,
        injected_models=args.injected_models
    )

    atexit.register(exit_handler, metadata_collector) # Program exit handler to save the metadata
    
    with open(output_path, 'w') as f:
        # sys.stdout = f  # Redirect stdout to the file
        # sys.stderr = f  # Redirect stderr to the file
        
        gcs_analysis_prefix = f"analysis_results/{TAG}/" 
        gcs_anomaly_prefix = f"anomalous_models/{TAG}/"
        gcs_malicious_prefix = f"malicious_models/{TAG}/"
        # gcs_injected_models_prefix = f"injected_models/"
        gcs_injected_models_prefix = f"injected_models_text-classification/"

        if args.retrieve_mal_models:
            prefix = gcs_malicious_prefix
        else:
            prefix = gcs_analysis_prefix    
        
        if UPLOAD_COMPONENT in args.active_components and not args.bucket_name:
            print("Please provide a Google Cloud Bucket name in which the data would be stored. Don't include 'upload' in active components if you don't want this functionality.")
        
        batch_number = 0
        if args.bucket_name:
            batch_number = (len(list_files_gcs(args.bucket_name, prefix)) + 1) if not args.wild_run else (len(metadata_collector.time_taken_df) + 1)
        num_iters = 0

        # Checking the checksum of the GCS versions provided the checksum of the HF version to see if there was any change in the pytorch_model.bin files.
        if args.checksum_verify:
            print_system_info()
            # Inialize the list of models in the folder
            all_models = list_files_gcs(args.bucket_name, gcs_analysis_prefix)
            models = [i for i in all_models if "/v2/" not in i]
            metadata_folder = os.path.join(args.work_dir, "metadata")
            checksum_log = os.path.join(metadata_folder, "checksum_verification_log.csv")
            archive_structure_log = os.path.join(metadata_folder, "old_ben_set_archive_structure.json")
            checksum_df = pd.DataFrame(columns=["name", "HF_checksum", "GCS_checksum", "matching", "corrupted"])
            archive_structures = {}
            checklist = ["gigabrain__cypto-tweets", "RajuKandasamy__tamillama_tiny_30m", "Ritori__Yura_GPT"]
            HF_checksums = {
                "gigabrain__cypto-tweets": "950da1308fbe458b12ec5caa3d18c7a9242f3c74a715b1a335605fefb64d42f5",
                "RajuKandasamy__tamillama_tiny_30m": "02b793d3d6f1d73841208077c361fb41b1d87cba8c957537ff61c69f884246a6",
                "Ritori__Yura_GPT": "486fc6f5c387628b002432ccf85ecd0fa23d29302747a6c59e42c0abf08d393f"
            }

            print(f"Found {len(models)} injected models in the GCS bucket.")

            # Continuing from last successful download
            start = 0
            # if last_success_download:
            #     for i, model in enumerate(models):
            #         model_name = os.path.basename(os.path.dirname(model))
            #         model_id = model_name.replace("__", "/", 1)
            #         if last_success_download == model_id:
            #             start = i + 1
            #             break
            for i in range(start, len(all_models)):
                model = all_models[i]
                
                if DOWNLOAD_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 1: Downloading models within space limit")
                    print("="*60)
                    
                    file_name = os.path.basename(model)

                    blob_path = f"{gcs_analysis_prefix}{file_name}"
                    destination_path = os.path.join(download_dir, file_name)
                    
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    
                    download_from_gcs(args.bucket_name, blob_path, destination_path)
                    
                    extracted = extract_analysis_archive(destination_path, os.path.dirname(download_dir))
                
                # Verify the checksum
                res = validate_folders(download_dir, checklist)

                if res:
                    for m_name in res:
                        checksum_df.loc[len(checksum_df)] = {
                            "name": m_name,
                            "HF_checksum": HF_checksums[m_name],
                            "GCS_checksum": res[m_name],
                            "matching": str(str(res[m_name]) == HF_checksums[m_name]),
                            "corrupted": str(not extracted)
                        }

                # Log the archive structure
                archive_model_list = get_folders_from_directory(download_dir)
                archive_structures[file_name] = archive_model_list + (["corrupted"] if not extracted else []) # To indicate whether the archive is corrupted/failed during extraction

                if CLEANUP_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 4: Cleaning up local files")
                    print("="*60)
                    
                    shutil.rmtree(download_dir)
                    print("Local files cleaned up")
                
                # Save all the metadata
                checksum_df.to_csv(checksum_log, index=False)
                with open(archive_structure_log, 'w') as outfile:
                    json.dump(archive_structures, outfile, indent=4)
                metadata_collector.save()

        if args.injected_models: # Retrieving the logs from GCS archived version of the injected models

            print_system_info()
            # Inialize the list of models in the folder
            all_models = list_files_gcs(args.bucket_name, gcs_injected_models_prefix)
            inj_model_name = "pytorch_model_injected_pypi.bin"
            models = [f for f in all_models if f.endswith(inj_model_name)] # Ensuring exact matching
            injected_set_path = os.path.join("metadata",'text-classification_injected.csv') 
            if not os.path.exists(injected_set_path):
                print(f"Unable to find the injected set path: {injected_set_path}")
                sys.exit(1)

            tc_models_check = set(pd.read_csv(injected_set_path)["name"])

            print(f"Found {len(models)} injected models in the GCS bucket.")

            # Continuing from last successful download
            start = 0
            # if last_success_download:
            #     for i, model in enumerate(models):
            #         model_name = os.path.basename(os.path.dirname(model))
            #         model_id = model_name.replace("__", "/", 1)
            #         if last_success_download == model_id:
            #             start = i + 1
            #             break
            for i in range(start, len(models)):
                model = models[i]
                
                if DOWNLOAD_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 1: Downloading models within space limit")
                    print("="*60)
                    
                    model_name = os.path.basename(os.path.dirname(model))
                    model_id = model_name.replace("__", "/", 1)
                    if model_id in explored_models:
                        print(f"Model {model_id} not present in the list. Skipping")
                        continue

                    blob_path = f"{gcs_injected_models_prefix}{model_name}/{inj_model_name}"
                    destination_path = os.path.join(download_dir, model_name, "pytorch_model.bin")
                    
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    
                    download_from_gcs(args.bucket_name, blob_path, destination_path)
                    
                    info = get_model_metadata_from_api(model_id)
                    injected_model_download_log_df.loc[len(injected_model_download_log_df)] = {
                        "name": model_id,
                        "filename": inj_model_name,
                        "likes": info.get("likes", 0),
                        "downloads": info.get("downloads", 0),
                        "last_updated": info.get("last_modified", "")
                    }
                    
                    injected_model_download_log_df.to_csv(injected_model_download_log, index=False)
                
                if DYNAMIC_ANALYSIS_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 2: Running dynamic analysis")
                    print("="*60)
                    
                    gc.collect()
                    run_malicious_samples(args.isModel, log_dir, out_dir, injected_models=True)
                    gc.collect()
                
                if CLASSIFICATION_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 3: Run Classifier to detect anomalies")
                    print("="*60)
                    
                    analyzer = StraceAnalyzer()
                    analysis_type = "model" if args.isModel else "dataset"
                    analyzer.load_data(
                        f"{log_dir}/malicious-straces/{analysis_type}/{TAG}/injected",
                        f"{log_dir}/clean-straces/{analysis_type}/{TAG}/bruh"
                    )
                    
                    if len(analyzer.benign_name_data) == 0 and len(analyzer.malicious_name_data) == 0:
                        print(f"Error: No data loaded. Please check your directory paths and file formats.")
                        sys.exit(1)
                        
                    analyzer.save_count_per_sample(mode="injected_models", tag=TAG)
                    
                    clf = SyscallAnomalyDetector()
                    clf.load_model(path=args.best_model_path)
                    test_data = pd.DataFrame(analyzer.malicious_name_data)
                    syscall_data = clf.extract_features(test_data, feature_types=args.feature_types)
                    predictions, dscores = clf.predict(syscall_data)
                    
                    for j in range(len(predictions)):
                        model_name = analyzer.malicious_name_data[j]["name"]
                        
                        if model_name in classified_set:
                            continue
                        
                        print(f"Model: {model_name}, Prediction: {'ANOMALY' if predictions[j] == -1 else 'BENIGN'}, Decision Score: {dscores[j]:.4f}")
                        
                        if predictions[j] == -1:
                            time.sleep(random.uniform(0.2, 0.75))
                            model_metadata = get_model_metadata_from_api(model_name)
                            
                            metadata_collector.log_anomaly({
                                "name": model_name,
                                "likes": model_metadata.get("likes", 0),
                                "downloads": model_metadata.get("downloads", 0),
                                "last_modified": model_metadata.get("last_modified", ""),
                                "decision_score": dscores[j]
                            })
                            
                            print(f"Anomaly detected in model: {model_name}")
                        else:
                            print(f"No anomaly detected in model: {model_name}")
                            
                        classified_set.add(model_name)
                
                if CLEANUP_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 4: Cleaning up local files")
                    print("="*60)
                    
                    shutil.rmtree(download_dir)
                    print("Local files cleaned up")
                
                metadata_collector.save()

        if args.run_inference: # Code for crawling HF or MALHUG local files.
            while (args.file_type in ["benign", "both"] or args.retrieve_mal_models) and len(set(download_df["name"])) < args.num_models:
                print(f"Number of models downloaded until now: {len(successful_downloaded_models)}")
                try:
                    # Download models with space constraints
                    print_system_info()
                    if args.wild_run or args.retrieve_mal_models:
                        if DOWNLOAD_COMPONENT in args.active_components:
                            print("\n" + "="*60)
                            print("STEP 1: Downloading models within space limit")
                            print("="*60)

                            metadata_collector.start_timer("download")
                            downloaded_models, num_iters, security_infos = download_models_with_space_limit(download_dir, space_limit_bytes, TAG, explored_models, wild_download_log_df if args.wild_run else download_log_df, args.num_models, args.mode, direction="asc", mal_run=args.retrieve_mal_models, last_success_download=last_success_download)
                            metadata_collector.stop_timer("download")
                            explored_models.update(downloaded_models)
                            last_success_download = downloaded_models[-1] if downloaded_models else last_success_download # update the last successful download

                        if DYNAMIC_ANALYSIS_COMPONENT in args.active_components:
                            # Run dynamic analysis
                            print("\n" + "="*60)
                            print("STEP 2: Running dynamic analysis")
                            print("="*60)
                            
                            gc.collect()
                            metadata_collector.start_timer("analysis")
                            if args.retrieve_mal_models:
                                print("Running dynamic analysis on malicious samples")
                                run_malicious_samples(args.isModel, log_dir, out_dir, wild_run=args.retrieve_mal_models)
                            else:
                                print("Running dynamic analysis on clean samples")
                                run_clean_samples(args.isModel, log_dir, out_dir, wild_run=args.wild_run)
                            gc.collect()
                            metadata_collector.stop_timer("analysis")

                        if CLASSIFICATION_COMPONENT in args.active_components:
                            # Running the classfier to detect anomalies
                            print("\n" + "="*60)
                            print("STEP 3: Run Classifier to detect anomalies")
                            print("="*60)
                            
                            metadata_collector.start_timer("classification")
                            analyzer = StraceAnalyzer()
                            analysis_type = "model" if args.isModel else "dataset"
                            analyzer.load_data(f"{log_dir}/malicious-straces/{analysis_type}/{TAG}/wild", f"{log_dir}/clean-straces/{analysis_type}/{TAG}/bruh")
                            if len(analyzer.benign_name_data) == 0 and len(analyzer.malicious_name_data) == 0:
                                print(f"Error: No data loaded. Please check your directory paths and file formats.")
                                sys.exit(1)
                            if not args.retrieve_mal_models:
                                analyzer.save_count_per_sample(mode="wild_run", tag=TAG)

                            clf = SyscallAnomalyDetector()
                            clf.load_model(path=args.best_model_path)
                            if args.retrieve_mal_models:
                                test_data = pd.DataFrame(analyzer.malicious_name_data)
                            else:
                                test_data = pd.DataFrame(analyzer.benign_name_data)
                            syscall_data = clf.extract_features(test_data, feature_types=args.feature_types)
                            predictions, dscores = clf.predict(syscall_data)

                            for i in range(len(predictions)):
                                if args.retrieve_mal_models:
                                    model_name = analyzer.malicious_name_data[i]["name"]
                                else:
                                    model_name = analyzer.benign_name_data[i]["name"]

                                if model_name in classified_set:
                                    continue

                                if predictions[i] == -1 or args.retrieve_mal_models:
                                    time.sleep(random.uniform(0.2, 0.75))  # Random delay to avoid rate limiting
                                    model_metadata = get_model_metadata_from_api(model_name)
                                    
                                    if args.wild_run:
                                        metadata_collector.log_anomaly({
                                            "name": model_name,
                                            "likes": model_metadata.get("likes", 0),
                                            "downloads": model_metadata.get("downloads", 0),
                                            "last_modified": model_metadata.get("last_modified", ""),
                                            "decision_score": dscores[i]
                                        })

                                    elif args.retrieve_mal_models:
                                        metadata_collector.log_wild_mal_models({
                                            "name": model_name,
                                            "likes": model_metadata.get("likes", 0),
                                            "downloads": model_metadata.get("downloads", 0),
                                            "last_updated": model_metadata.get("last_modified", ""),
                                            **security_infos.get(model_name, {}),
                                            "clf_prediction": "ANOMALY" if predictions[i] == -1 else "BENIGN",
                                            "clf_decision_score": dscores[i]
                                        })

                                    print(f"Anomaly detected in model: {model_name}")
                                else:
                                    print(f"No anomaly detected in model: {analyzer.benign_name_data[i]['name']}")
                                classified_set.add(model_name)
                            metadata_collector.stop_timer("classification")
                        upload_success = True
                        if UPLOAD_COMPONENT in args.active_components:
                            if not args.retrieve_mal_models:
                                metadata_collector.start_timer("upload")
                                # Upload to GCS
                                print("\n" + "="*60)
                                print("STEP 4: Uploading anomaly models to Google Cloud Storage")
                                print("="*60)
                                
                                for model_id in anomalous_df["name"].tolist():
                                    model_name = model_id.replace("/", "__")        
                                    
                                    if model_id in uploaded_set: # Skip already uploaded models
                                        print(f"Model {model_id} already uploaded. Skipping...")
                                        continue  

                                    model_file_path = os.path.join(download_dir, model_name, "pytorch_model.bin")
                                    blob_name = f"{gcs_anomaly_prefix}{clf_name}/{model_name}/pytorch_model.bin"
                                    upload_success = upload_to_gcs(
                                        model_file_path, 
                                        args.bucket_name, 
                                        blob_name
                                    )
                                metadata_collector.stop_timer("upload")
                            else:
                                # ZIP the model files for archiving
                                print("\n" + "="*60)
                                print("STEP 3: Creating analysis archive")
                                print("="*60)
                                
                                output_archive = os.path.join(download_dir, f"analysis_archive_{TAG}_batch_{batch_number}.tar.zst")
                                archive_path = os.path.join(work_dir, output_archive)
                                create_analysis_archive(download_dir, archive_path)
                                
                                # Upload to GCS
                                print("\n" + "="*60)
                                print("STEP 4: Uploading to Google Cloud Storage")
                                print("="*60)
                                
                                blob_name = f"{gcs_malicious_prefix}{clf_name}/{os.path.basename(output_archive)}"
                                upload_success = upload_to_gcs(
                                    archive_path, 
                                    args.bucket_name, 
                                    blob_name
                                )

                            if upload_success:
                                print("Pipeline completed successfully!")
                                uploaded_set.update(anomalous_df["name"].tolist())
                            else:
                                print("Upload failed. Check your GCS configuration.")

                        if CLEANUP_COMPONENT in args.active_components:
                            print("\n" + "="*60)
                            print("STEP 5: Cleaning up local files")
                            print("="*60)
                            
                            if upload_success:
                                batch_number += 1
                            shutil.rmtree(download_dir)

                            print("Local files cleaned up")

                        metadata_collector.log_batch_times({
                            "batch": batch_number,
                            "batch_size": len(downloaded_models)
                        })

                        metadata_collector.save()

                        total_time = sum(v for k, v in metadata_collector.times.items() if 'wall' in k)
                        print(f"Time taken for running dynamic analysis on {len(downloaded_models)} models: {total_time:.2f} seconds")
                    else:
                        if DOWNLOAD_COMPONENT in args.active_components:

                            print("\n" + "="*60)
                            print("STEP 1: Downloading models within space limit")
                            print("="*60)
                            
                            gc.collect()
                            downloaded_models, num_iters, _ = download_models_with_space_limit(download_dir, space_limit_bytes, TAG, explored_models, download_log_df, args.num_models, args.mode, last_success_download=last_success_download)
                        
                            # if not downloaded_models:
                            #     print("No models downloaded. Exiting.")
                            #     return
                        
                            explored_models.update(downloaded_models)
                            last_success_download = downloaded_models[-1] if downloaded_models else last_success_download

                        if DYNAMIC_ANALYSIS_COMPONENT in args.active_components:
                            # Run dynamic analysis
                            print("\n" + "="*60)
                            print("STEP 2: Running dynamic analysis")
                            print("="*60)
                            
                            gc.collect()
                            run_clean_samples(args.isModel, log_dir, out_dir)
                            gc.collect()

                        # For removing already uploaded files from the download directory.
                        if REMOVE_EXPLORED in args.active_components: 
                            print("=" * 60)
                            print("Removing explored model files")
                            print("=" * 60)
                            filtered_df = old_download_log_df[old_download_log_df["name"].isin(downloaded_models)].copy()

                            for i, row in filtered_df.iterrows():
                                model_id, filename = row["name"], row["filename"]
                                local_dir = os.path.join(download_dir, model_id.replace("/", "__"))
                                remove_file_from_directory(local_dir, filename)

                        upload_success = True
                        if UPLOAD_COMPONENT in args.active_components:
                            # ZIP the model files for archiving
                            print("\n" + "="*60)
                            print("STEP 3: Creating analysis archive")
                            print("="*60)
                            
                            output_archive = os.path.join(download_dir, f"analysis_archive_{TAG}_batch_{batch_number}.tar.zst")
                            archive_path = os.path.join(work_dir, output_archive)
                            create_analysis_archive(download_dir, archive_path)
                            
                            # Upload to GCS
                            print("\n" + "="*60)
                            print("STEP 4: Uploading to Google Cloud Storage")
                            print("="*60)
                            
                            blob_name = f"{gcs_analysis_prefix}{os.path.basename(output_archive)}"
                            upload_success = upload_to_gcs(
                                archive_path, 
                                args.bucket_name, 
                                blob_name
                            )
                            
                            if upload_success:
                                print("Pipeline completed successfully!")
                            
                            else:
                                print("Upload failed. Check your GCS configuration.")

                        # Step 5: Cleanup (optional)
                        if CLEANUP_COMPONENT in args.active_components:
                            print("\n" + "="*60)
                            print("STEP 5: Cleaning up local files")
                            print("="*60)
                            
                            if upload_success:
                                batch_number += 1
                            else:
                                print("Skipping cleanup due to upload failure.")
                                continue
                            shutil.rmtree(download_dir)
                            print("Local files cleaned up")
                except Exception as e:
                    print(f"Pipeline failed with error: {e}")
                    traceback.print_exc()
                    sys.exit(1)

            if args.file_type in ["malicious", "both"] and not args.retrieve_mal_models:
                print("\n" + "="*60)
                print("Running dynamic analysis on malicious samples")
                print("="*60)
                
                run_malicious_samples(args.isModel, log_dir, out_dir)
        
        # Crawl HF with a pre-determined list of models to download
        if args.download_from_csv:
            old_downloaded_models = pd.read_csv("data/clean_dataset/text-classification_downloaded_models.csv")["name"]
            i = 0
            while len(set(download_df["name"])) < args.num_models and i < len(old_downloaded_models):
                model_id = old_downloaded_models[i]
                if DOWNLOAD_COMPONENT in args.active_components:

                    print("\n" + "="*60)
                    print("STEP 1: Downloading models within space limit")
                    print("="*60)
                    
                    downloaded_models, num_iters, _ = download_from_old_list(model_id, download_dir, space_limit_bytes, TAG, download_log_df, args.mode, last_success_download=last_success_download)
                    # if not downloaded_models:
                    #     print("No models downloaded. Exiting.")
                    #     return

                    if not downloaded_models:
                        i += 1
                        continue

                    explored_models.update(downloaded_models)
                    last_success_download = downloaded_models[-1] if downloaded_models else last_success_download

                if DYNAMIC_ANALYSIS_COMPONENT in args.active_components:
                    # Run dynamic analysis
                    print("\n" + "="*60)
                    print("STEP 2: Running dynamic analysis")
                    print("="*60)
                    
                    gc.collect()
                    run_clean_samples(args.isModel, log_dir, out_dir)
                    gc.collect()

                # Step 5: Cleanup (optional)
                upload_success = True
                if CLEANUP_COMPONENT in args.active_components:
                    print("\n" + "="*60)
                    print("STEP 5: Cleaning up local files")
                    print("="*60)
                    
                    if upload_success:
                        batch_number += 1
                    else:
                        print("Skipping cleanup due to upload failure.")
                        continue
                    shutil.rmtree(download_dir)
                    print("Local files cleaned up")
                i += 1

        # Run the strace analyzer on the logs collected.
        print("Running strace analyzer on the logs collected...") 
        analyzer = StraceAnalyzer()
        analysis_type = "model" if args.isModel else "dataset"
        analyzer.load_data(f"{log_dir}/malicious-straces/{analysis_type}/{TAG}", f"{log_dir}/clean-straces/{analysis_type}/{TAG}")
        if len(analyzer.malicious_name_data) == 0 and len(analyzer.benign_name_data) == 0:
            print(f"Error: No data loaded. Please check your directory paths and file formats.")
            sys.exit(1)
            
        print("Dynamic analysis completed.")

        if args.save_count_per_sample:
            analyzer.save_count_per_sample(tag=TAG, mode="wild_run" if args.wild_run else "normal_run")
if __name__ == "__main__":
    main()