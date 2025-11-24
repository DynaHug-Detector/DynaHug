# DynaHug Experiment Scripts

This is the collection of scripts used for experiments involving the machine-learning model with static features and malicious-payload injection. We also provide scripts that were used for data analysis of our datasets.

# Scripts Provided

- `main.py` - Runs the pipeline for generating opcodes, processing them and combining them

- `fickle_script.py` - Script to disassemble model files in a directory

- `opcode_analysis.py` - Non-generalised feature processing script

- `generalise_opcodes.py` - Generalised feature processing script

- `pytorch_injector.py` - Inject a pytorch model with a custom generated payload

- `download_hf_models.py` - Download script for models from HuggingFace

- `download_gcs.py` - Download script for models on Google Cloud Storage

- `opensource_runner.py` - Run baseline opensource tools on a file. NOTE: Please follow install instructions at [ModelTracer's Repo for strace2csv](https://github.com/s2e-lab/hf-model-analyzer/tree/main?tab=readme-ov-file). Also pay attention to pyenv flag.

# Instructions To Run `main.py`

`main.py` runs the full pipeline of static analysis, encompassing opcode-generation -> opcode analysis -> opcode selection. Below is a brief description for activating each of the steps:

- `--base-dir` Directory containing the models that are being used for the analysis. Required flag

- `--generate` Flag to enable fickling generation for models in directory static analysis. This is the longest step in the pipeline

- `--filtering` Use if there exists a csv to filter the names by

- `--csv-path` Use to point to the csv that the analysis is being filtered by

- `--ngrams` Use if only Non-generalised features are desired

- `--malhug` To indicate if the target directory is MalHug

# Example of running `main.py`

```bash
python main.py --base-dir /mnt/The_Second_Drive/Security/ML_Research/downloaded_injected/ --generate 
```

# Auxiliary Scripts

- `malhug_pickleball_comparer.py` - Compare malicious sets of MalHug and Pickleball

- `pickle_opcodes.py` - Generate a list of pickle opcodes

- `payload_generator.py` = Generate a payload for injection. Inspired from [coldwater_q's injector](https://github.com/coldwaterq/pickle_injector)

- `opcode_combiner.py` - Combine n_grams to make one opcode list

- `fickling_safety.py` - Run Fickling's safety script on a directory

- `cluster_checker.py` - Analyse HuggingFace for clusters

- `extract_used.py` - Extraction script for MalHug

- `fickle_insertion.py` - Script for injection with Fickling

- `model_tracer_runner.py` - Script for calling ModelTracer from a file

- `most_used.py` - Script to generate statistics for MalHug models

- `utils.py` - Utility functions for download scripts
