from collections import defaultdict
from pathlib import Path
import traceback
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
import os
import joblib
import argparse
import sys
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score
)
import shap
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # to import utils from parent directory
from utils.utils import parse_feature_from_file
import matplotlib.pyplot as plt

class SyscallAnomalyDetector:
    def __init__(self, model_path="./classifier/models/", classifier_class=OneClassSVM, classifier_params={"nu": 0.05, "gamma": "scale", "kernel": "rbf"}, tag=None, hybrid=False):
        self.model_path = model_path
        self.vectorizer = None
        self.scaler = None
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.model = None
        self.syscall_names = None
        self.tag = tag
        self.syscall_file = os.path.join("classifier", "syscalls.txt")
        self.opcode_file = os.path.join("classifier", "opcodes.txt")
        self.hybrid = hybrid
        
        os.makedirs(model_path, exist_ok=True)
    
    def extract_features(self, data, feature_types=["presence"]):
        """
        Extract features from system call data based on specified feature types.
        In Hybrid mode, normal syscall features and opcode features are concatenated.
        proc_seq_ -> process sequence features or generalized sequence opcode features.
        seq_ -> sequence features for syscall/opcodes.
        gen_ -> generalized features for opcodes
        gen_seq_ -> generalized sequence features for opcodes
        """
        self.syscall_names = parse_feature_from_file(self.syscall_file)

        if self.hybrid:
            self.syscall_names += parse_feature_from_file(self.opcode_file) # concatenate syscall and opcode names  
        if data.dropna(how='all').empty:
            print("Error: The data passed was found to be empty.")
            sys.exit(1)

        feature_dicts = []
        for _, row in data.iterrows():
            feature_dict = {}
            
            # Common static and dynamic features
            for syscall in self.syscall_names:
                value = 0 if syscall not in row or pd.isna(row[syscall]) else row[syscall]
                
                if "presence" in feature_types:
                    feature_dict[f"presence_{syscall}"] = 1 if value > 0 else 0
                    
                if "frequency" in feature_types:
                    feature_dict[f"frequency_{syscall}"] = value

            cols = [col for col in row.index if col.startswith('seq_')]
            if ("sequence_presence" in feature_types or "sequence_frequency" in feature_types) and not cols:
                print("Error: No sequence features found!")
                sys.exit(1)
            for col in cols:    
                if "sequence_presence" in feature_types:
                    if pd.notna(row[col]) and row[col] > 0:
                        feature_dict[f"presence_{col}"] = 1
                    else:
                        feature_dict[f"presence_{col}"] = 0

                if "sequence_frequency" in feature_types:
                    if pd.notna(row[col]):
                        feature_dict[f"frequency_{col}"] = row[col]

            # Dynamic features
            cols = [col for col in row.index if col.startswith('proc_seq_')]
            if ("proc_seq_presence" in feature_types or "proc_seq_frequency" in feature_types) and not cols:
                print("Error: No process sequence features found!")
                sys.exit(1)

            for col in cols:
                if "proc_seq_presence" in feature_types:
                    if pd.notna(row[col]) and row[col] > 0:
                        feature_dict[f"presence_{col}"] = 1
                    else:
                        feature_dict[f"presence_{col}"] = 0

                if "proc_seq_frequency" in feature_types:
                    if pd.notna(row[col]):
                        feature_dict[f"frequency_{col}"] = row[col]

            # Static features
            cols = [col for col in row.index if col.startswith('gen_')]
            if ("gen_presence" in feature_types or "gen_frequency" in feature_types) and not cols:
                print("Error: No generalized opcode features found!")
                sys.exit(1)
            for col in cols:
                if "gen_presence" in feature_types:
                    if pd.notna(row[col]) and row[col] > 0:
                        feature_dict[f"presence_{col}"] = 1
                    else:
                        feature_dict[f"presence_{col}"] = 0

                if "gen_frequency" in feature_types:
                    if pd.notna(row[col]):
                        feature_dict[f"frequency_{col}"] = row[col]
            
            cols = [col for col in row.index if col.startswith('gen_seq_')]
            if ("gen_seq_presence" in feature_types or "gen_seq_frequency" in feature_types) and not cols:
                print("Error: No generalized sequence opcode features found!")
                sys.exit(1)
            for col in cols:
                if "gen_seq_presence" in feature_types:
                    if pd.notna(row[col]) and row[col] > 0:
                        feature_dict[f"presence_{col}"] = 1
                    else:
                        feature_dict[f"presence_{col}"] = 0

                if "gen_seq_frequency" in feature_types:
                    if pd.notna(row[col]):
                        feature_dict[f"frequency_{col}"] = row[col]

            feature_dicts.append(feature_dict)
        
        return feature_dicts

    def instantiate_model(self, params=None):
        """
        Handles instantiation of the model based on the classifier class and parameters
        """
        final_params = params if params else self.classifier_params
        if self.classifier_class == OneClassSVM:
            return OneClassSVM(
                kernel=final_params.get("kernel", "rbf"),
                gamma=final_params.get("gamma", "scale"),
                nu=final_params.get("nu", 0.05)
            )
        elif self.classifier_class == SGDOneClassSVM:
            return make_pipeline(
                Nystroem(kernel=final_params.get("kernel", "rbf"), gamma=final_params.get("gamma", "scale")),
                SGDOneClassSVM(nu=final_params.get("nu", 0.05))
            )
        elif self.classifier_class == IsolationForest:
            return IsolationForest(
                n_estimators=final_params.get("n_estimators", 100),
                max_samples=final_params.get("max_samples", "auto"),
                contamination=final_params.get("contamination", "auto"),
                random_state=final_params.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unsupported classifier class: {self.classifier_class}")
            
    def predict(self, syscall_data):
        """
        Predict whether syscall patterns are normal or anomalous
        """
        if self.model is None:
            self.load_model(self.model_path)

        X_counts = self.vectorizer.transform(syscall_data)
        
        feature_names = self.vectorizer.get_feature_names_out()
        frequency_indices = [i for i, name in enumerate(feature_names) if name.startswith('frequency_')] 
        
        X_scaled = X_counts.copy()
        if frequency_indices:
            X_scaled[:, frequency_indices] = self.scaler.transform(X_counts[:, frequency_indices])
        
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        return predictions, scores
    
    def save_model(self, path):
        """Save the trained model and preprocessing components"""
        model_file = os.path.join(path, "oneclass_svm_model.pkl")
        vectorizer_file = os.path.join(path, "vectorizer.pkl")
        scaler_file = os.path.join(path, "scaler.pkl")
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.vectorizer, vectorizer_file)
        joblib.dump(self.scaler, scaler_file)
        
        # Save syscall names for reference
        with open(os.path.join(self.model_path, "syscall_names.txt"), "w") as f:
            f.write("\n".join(self.syscall_names))
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self, path):
        """Load the trained model and preprocessing components"""
        model_file = os.path.join(path, "oneclass_svm_model.pkl")
        vectorizer_file = os.path.join(path, "vectorizer.pkl")
        scaler_file = os.path.join(path, "scaler.pkl")
        
        if not all(os.path.exists(f) for f in [model_file, vectorizer_file, scaler_file]):
            raise FileNotFoundError("Model files not found. Make sure you gave the right path.")
        
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(vectorizer_file)
        self.scaler = joblib.load(scaler_file)
        
        # Load syscall names
        syscall_names_file = os.path.join(self.model_path, "syscall_names.txt")
        if os.path.exists(syscall_names_file):
            with open(syscall_names_file, "r") as f:
                self.syscall_names = f.read().strip().split("\n")
        
        print("Model loaded successfully!")
    
    def obtain_false_positives(self, data, y_pred, y_true):
        """
        Obtain false positive samples from benign data based on predictions
        """
        false_positives = data[(y_pred == -1) & (y_true == 1)]
        if "name" in false_positives.columns:
            return false_positives[["name"]]
        else:
            print("Something is wrong, no 'name' column found in false positives.")
            print("Available columns:", false_positives.columns)
            sys.exit(1)

    def analyze_samples(self, samples, y_true, feature_types):
        """
        Analyze the performance of the model on a set of samples(benign or malicious)
        """
        test_syscall_data = self.extract_features(samples, feature_types=feature_types)
        predictions, scores = self.predict(test_syscall_data)

        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[1, -1]).ravel().tolist()
        
        fp_rows = self.obtain_false_positives(samples, predictions, y_true)

        accuracy = (tp + tn) / (tp + tn + fp + fn) 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # True Positives / (True Positives + False Positives)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # True Positives / (True Positives + False Negatives)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # Harmonic mean of precision and recall

        y_true_roc = np.array([1 if label == 1 else 0 for label in y_true])  # Convert -1 to 0 and vice versa for ROC AUC calculation
        roc_auc = roc_auc_score(y_true_roc, scores) if len(set(y_true)) > 1 else None

        return predictions, tp, tn, fp, fn, accuracy, precision, recall, f1, roc_auc, fp_rows

    def perform_k_fold(self, ben_data, mal_data, feature_types, params=None, k=5, mode=""):
        """
        k-fold for cross validation of performance of the model
        """
        results = pd.DataFrame(columns=['fold', 'accuracy', 'precision', 'recall', 'f1', 'tp', 'tn', 'fp', 'fn', 'roc_auc', 'total_samples'])
        best_fold_model = None
        best_fold_score = 0
        best_fold_dir = None
        best_fold_index = -1

        # Shuffle the training data 
        ben_data = ben_data.sample(frac=1, random_state=42).reset_index(drop=True)
        mal_data = mal_data.sample(frac=1, random_state=42).reset_index(drop=True)

        fold_size = len(ben_data) // k
        mal_size = len(mal_data)
        feature_type_string = "_".join(feature_types)
        hybrid_suffix = "_hybrid" if self.hybrid else "" + f"_{mode}" if mode != "" else ""
        model_save_dir = os.path.join(self.model_path, f"{len(ben_data)}_benign_data_{feature_type_string}{hybrid_suffix}", self.classifier_class.__name__)
        
        X_mal_val = mal_data[:mal_size // 2]
        X_mal_test = mal_data[mal_size // 2:]

        for fold in range(k):
            print(f"\n=== K-Fold Iteration {fold + 1} for params {params} ===")
            val_start = fold * fold_size
            val_end = val_start + fold_size // 2  # Half of the fold size for validation
            test_start = val_end
            test_end = val_start + fold_size if fold < k - 1 else len(ben_data)  # Special handling for last fold
            
            X_train = pd.concat([ben_data[:val_start], ben_data[test_end:]])
            X_ben_val = ben_data[val_start:val_end]
            X_ben_test = ben_data[test_start:test_end]
            
            X_val = pd.concat([X_ben_val, X_mal_val])
            X_test = pd.concat([X_ben_test, X_mal_test])
            
            y_true_ben_val = np.ones(len(X_ben_val))  # Benign samples are labeled as 1
            y_true_mal_val = -1 * np.ones(len(X_mal_val))  # Malicious samples are labeled as -1
            y_true_val = np.concatenate([y_true_ben_val, y_true_mal_val])
            
            y_true_ben_test = np.ones(len(X_ben_test))
            y_true_mal_test = -1 * np.ones(len(X_mal_test))
            y_true_test = np.concatenate([y_true_ben_test, y_true_mal_test])

            print(f"Training the {self.classifier_class.__name__}")
            self.vectorizer = DictVectorizer(sparse=False)
            # self.scaler = StandardScaler()
            self.scaler = StandardScaler(with_mean=False)
            # self.scaler = MaxAbsScaler()
            print(f"Scaler being used: {type(self.scaler).__name__}")
            print(f"Vectorizer being used: {type(self.vectorizer).__name__}")

            self.model = self.instantiate_model(params=params)

            if isinstance(self.model, Pipeline):
                # If the model is an instance of SGDOneClassSVM with Nystroem, we need to use batch training
                print("Using batch training for SGDOneClassSVM with Nystroem")
                batch_size = self.classifier_params.get("batch_size", 128)
                
                transformer = self.model.steps[0][1] # Nystroem
                estimator = self.model.steps[1][1]   # SGDOneClassSVM

                first_batch_df = X_train.iloc[0:batch_size]
                syscall_dicts_first_batch = self.extract_features(first_batch_df, feature_types=feature_types)
                X_counts_first_batch = self.vectorizer.fit_transform(syscall_dicts_first_batch)
                
                frequency_indices = [i for i, name in enumerate(self.vectorizer.get_feature_names_out()) if name.startswith('frequency_')]
                if frequency_indices:
                    self.scaler.fit(X_counts_first_batch[:, frequency_indices])

                for i in range(0, len(X_train), batch_size):
                    batch_df = X_train.iloc[i:i+batch_size]
                    syscall_dicts_batch = self.extract_features(batch_df, feature_types=feature_types)
                    X_counts_batch = self.vectorizer.transform(syscall_dicts_batch)
                    X_scaled_batch = X_counts_batch.copy()
                    if frequency_indices:
                        X_scaled_batch[:, frequency_indices] = self.scaler.transform(X_counts_batch[:, frequency_indices])
                    
                    if i == 0:
                        transformed_batch = transformer.fit_transform(X_scaled_batch)
                    else:
                        transformed_batch = transformer.transform(X_scaled_batch)
                    
                    estimator.partial_fit(transformed_batch)
            else:
                syscall_dicts = self.extract_features(X_train, feature_types=feature_types)
                X_counts = self.vectorizer.fit_transform(syscall_dicts)
                X_scaled = X_counts.copy()

                feature_names = self.vectorizer.get_feature_names_out()
                frequency_indices = [i for i, name in enumerate(feature_names) if name.startswith('frequency_')]

                if frequency_indices:
                    X_scaled[:, frequency_indices] = self.scaler.fit_transform(X_counts[:, frequency_indices])
                self.model.fit(X_scaled)

            print("Extracted features:", self.vectorizer.get_feature_names_out())
            fold_dir = os.path.join(model_save_dir, f"fold_{fold + 1}")
            os.makedirs(fold_dir, exist_ok=True)
            self.save_model(fold_dir)
            print("Training completed successfully!")

            # Analyze on validation set 
            _, tp, tn, fp, fn, accuracy, precision, recall, f1, roc_auc, fp_data = self.analyze_samples(X_val, y_true_val, feature_types=feature_types)

            results.loc[len(results)] = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'roc_auc': roc_auc,
                'total_samples': len(X_val)
            }

            print("Validation metrics:")
            print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc}")
            print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} (total {len(X_val)})")
            print(f"  False-positive sample count (from this fold): {len(fp_data)}")
            if hasattr(fp_data, "empty") and not fp_data.empty:
                print("  Example FP rows:")
                try:
                    print(fp_data.head(5))
                except Exception:
                    print("  (unable to pretty-print fp_data)")
            
            # Tracking the best model
            if f1 > best_fold_score:
                best_fold_score = f1
                best_fold_dir = fold_dir
                best_fold_index = fold

                best_fold_model = {
                    'model': joblib.load(os.path.join(fold_dir, "oneclass_svm_model.pkl")),
                    'vectorizer': joblib.load(os.path.join(fold_dir, "vectorizer.pkl")),
                    'scaler': joblib.load(os.path.join(fold_dir, "scaler.pkl")),
                    'test_data': X_test,
                    'test_labels': y_true_test,
                    'fp_samples': fp_data,
                    'test_data_start_idx': val_end,
                    'test_data_end_idx': test_end
                }

                print("\nMore information on test set in the best fold model:")
                print(f"Test set starts from {best_fold_model['test_data_start_idx']}")
                print(f"Test set ends at {best_fold_model['test_data_end_idx']}")

        print(f"\nBest fold: {best_fold_index + 1} with F1 score: {best_fold_score:.4f}")
        return results, best_fold_model, best_fold_dir

    def setup_train_dataset(self):
        # text-generation dynamic
        BENIGN_SYSCALL_DATA_tg = "classifier/data/new_benign_syscall_counts_text-generation.parquet"
        MALICIOUS_SYSCALL_DATA_tg = "classifier/data/new_malicious_syscall_counts_text-generation.parquet"
        DYNAMIC_INJECTED_DATA_tg = "classifier/data/injected_malicious_syscall_counts_text-generation.parquet"

        # text-generation static
        BENIGN_OPCODE_DATA_tg = "classifier/data/static-3000-text-generation-benign.csv"
        MALICIOUS_OPCODE_DATA_tg = "classifier/data/malicious_malhug_opcode_text_generation_all_feats_counts.csv"
        STATIC_INJECTED_DATA_tg = "classifier/data/injected-mal-opcodes.csv"

        # text-classification dynamic
        BENIGN_SYSCALL_DATA_tc = "classifier/data/new_benign_syscall_counts_text-classification.parquet"
        MALICIOUS_SYSCALL_DATA_tc = "classifier/data/new_MALHUG_tc_malicious_syscall_counts_text-classification.parquet"
        DYNAMIC_INJECTED_DATA_tc = "classifier/data/new_MALHUG_injected_malicious_syscall_counts_text-classification.parquet"

        # text-classification static (in progress)
        BENIGN_OPCODE_DATA_tc = None
        MALICIOUS_OPCODE_DATA_tc = None
        STATIC_INJECTED_DATA_tc = None

        # non-clustered dynamic
        BENIGN_SYSCALL_DATA_all = "classifier/data/new_benign_syscall_counts_all.parquet"
        MALICIOUS_SYSCALL_DATA_all = "classifier/data/new_MALHUG_all_malicious_syscall_counts_all.parquet"
        DYNAMIC_INJECTED_DATA_all = "classifier/data/new_MALHUG_injected_malicious_syscall_counts_all.parquet"

        # non-clustered static (in progress)
        BENIGN_OPCODE_DATA_all = None
        MALICIOUS_OPCODE_DATA_all = None
        STATIC_INJECTED_DATA_all = None

        DATASET_MAP = {
            "text-generation": (BENIGN_SYSCALL_DATA_tg, BENIGN_OPCODE_DATA_tg, MALICIOUS_SYSCALL_DATA_tg, MALICIOUS_OPCODE_DATA_tg, DYNAMIC_INJECTED_DATA_tg, STATIC_INJECTED_DATA_tg),
            "text-classification":(BENIGN_SYSCALL_DATA_tc, BENIGN_OPCODE_DATA_tc, MALICIOUS_SYSCALL_DATA_tc, MALICIOUS_OPCODE_DATA_tc, DYNAMIC_INJECTED_DATA_tc, STATIC_INJECTED_DATA_tc),
            "all": (BENIGN_SYSCALL_DATA_all, BENIGN_OPCODE_DATA_all, MALICIOUS_SYSCALL_DATA_all, MALICIOUS_OPCODE_DATA_all, DYNAMIC_INJECTED_DATA_all, STATIC_INJECTED_DATA_all)
        }

        return DATASET_MAP

    def setup_eval_dataset(self, analysis_type):
        """
        Function to setup the evaluation dataset based on analysis type and tags.
        """

        # Evaluation dataset 
        try:
            # Text-generation dynamic
            malhug_tg_dynamic_data = pd.read_parquet('classifier/data/new_malicious_syscall_counts_text-generation.parquet')
            pypi_inj_dynamic_data_tg = pd.read_parquet("classifier/data/new_pypi_injected_malicious_syscall_counts_text-generation.parquet") # PyPI injected set
            malhug_inj_dynamic_data_tg = pd.read_parquet('classifier/data/new_MALHUG_injected_malicious_syscall_counts_text-generation.parquet').head(1000)
            pklball_mal_set_dynamic_data_tg = pd.read_csv("classifier/data/new_pklball_mal_malicious_syscall_counts_text-generation.csv") # text generation pickleball models
            hf_mal_wild_dynamic_data_tg = pd.read_csv("classifier/data/new_HF_wild_mal_text-generation.csv") # HF Malicious wild dynamic
            hf_ben_test_set_dynamic_data_tg =  pd.read_parquet("classifier/data/new_2025_test_set_benign_syscall_counts_text-generation.parquet") # Loading test set data from the training set

            malhug_tg_data = malhug_tg_dynamic_data
            pklball_mal_data_tg = pklball_mal_set_dynamic_data_tg
            hf_mal_wild_data_tg = hf_mal_wild_dynamic_data_tg
            hf_ben_test_set_data_tg = hf_ben_test_set_dynamic_data_tg
            malhug_inj_data_tg = malhug_inj_dynamic_data_tg
            pypi_inj_data_tg = pypi_inj_dynamic_data_tg

            malicious_real_data_tg = pd.concat([malhug_tg_dynamic_data, pklball_mal_set_dynamic_data_tg, hf_mal_wild_dynamic_data_tg])
            print(f"Malicious real data size for text-generation: {len(malicious_real_data_tg)}")

            # Text-generation static
            if analysis_type == "static" or analysis_type == "hybrid":
                malhug_tg_static_data = pd.read_csv('classifier/data/malicious_malhug_opcode_text_generation_all_feats_counts.csv')
                pypi_inj_static_data = pd.read_csv("classifier/data/pypi_inj_set_static_opcodes.csv")
                malhug_inj_static_data = pd.read_csv("classifier/data/injected_opcodes.csv") # MALHUG injected opcodes
                hf_mal_wild_static_data = pd.read_csv("classifier/data/HF_malicious_wild_opcodes.csv") # Malicious in the wild opcodes
                hf_ben_test_set_static_data = pd.read_csv("classifier/data/2025_benign_test_opcodes.csv") 
                pklball_mal_set_static_data = pd.read_csv("classifier/data/Pickleball-malicious-opcodes.csv")
                
                if analysis_type == "static":
                    malhug_tg_data = malhug_tg_static_data
                    pypi_inj_data_tg = pypi_inj_static_data
                    malhug_inj_data_tg = malhug_inj_static_data
                    hf_mal_wild_data_tg = hf_mal_wild_static_data
                    pklball_mal_data_tg = pklball_mal_set_static_data
                    hf_ben_test_set_data_tg = hf_ben_test_set_static_data

                # Text-generation hybrid
                if analysis_type == "hybrid":
                    malhug_tg_data = pd.merge(malhug_tg_dynamic_data, malhug_tg_static_data, on=["name", "filename"])
                    malhug_inj_data_tg = pd.merge(malhug_inj_dynamic_data_tg, malhug_inj_static_data, on=["name"])
                    pklball_mal_data_tg = pd.merge(pklball_mal_set_dynamic_data_tg, pklball_mal_set_static_data, on=["name", "filename"])
                    pypi_inj_data_tg = pd.merge(pypi_inj_dynamic_data_tg, pypi_inj_static_data, on=["name"])
                    hf_mal_wild_data_tg = pd.merge(hf_mal_wild_dynamic_data_tg, hf_mal_wild_static_data, on=["name"])
                    hf_ben_test_set_data_tg = pd.merge(hf_ben_test_set_dynamic_data_tg, hf_ben_test_set_static_data, on=["name"])

            # Text-classification dynamic
            malhug_tc_dynamic_data = pd.read_parquet('classifier/data/new_MALHUG_tc_malicious_syscall_counts_text-classification.parquet')
            pypi_inj_dynamic_data_tc = pd.read_parquet("classifier/data/new_pypi_injected_malicious_syscall_counts_text-classification.parquet").head(1000) # PyPI injected set
            malhug_inj_dynamic_data_tc = pd.read_parquet('classifier/data/new_MALHUG_injected_malicious_syscall_counts_text-classification.parquet').head(1000)
            pklball_mal_set_dynamic_data_tc = pd.read_parquet("classifier/data/new_pklball_malicious_syscall_counts_text-classification.parquet") # 3 gram text generation pickleball models
            hf_ben_test_set_dynamic_data_tc = pd.read_parquet("classifier/data/new_2025_test_set_benign_syscall_counts_text-classification.parquet").head(2004) # Loading test set data from the training set

            malhug_tc_data = malhug_tc_dynamic_data
            pklball_mal_data_tc = pklball_mal_set_dynamic_data_tc
            hf_ben_test_set_data_tc = hf_ben_test_set_dynamic_data_tc
            malhug_inj_data_tc = malhug_inj_dynamic_data_tc
            pypi_inj_data_tc = pypi_inj_dynamic_data_tc

            malicious_real_data_tc = pd.concat([malhug_tc_data, pklball_mal_set_dynamic_data_tc])
            print(f"Malicious real data size for text-classification: {len(malicious_real_data_tc)}")

            # text-classification static and hybrid data have not been processed yet.

            DATASET_MAP = {
                "text-generation": {
                    "HF_ben_test": hf_ben_test_set_data_tg,
                    "HF_wild_mal": hf_mal_wild_data_tg,
                    "injected_pypi": pypi_inj_data_tg,
                    "MALHUG_tg": malhug_tg_data,
                    "MALHUG_injected": malhug_inj_data_tg,
                    "pklball_mal": pklball_mal_data_tg,
                    "mal_real": malicious_real_data_tg
                },
                "text-classification": {
                    "HF_ben_test": hf_ben_test_set_data_tc,
                    "injected_pypi": pypi_inj_data_tc,
                    "MALHUG_tc": malhug_tc_data,
                    "MALHUG_injected": malhug_inj_data_tc,
                    "pklball_mal": pklball_mal_data_tc,
                    "mal_real": malicious_real_data_tc
                }
            }

            return DATASET_MAP
        except Exception as e:
            traceback.print_exc()
            return None    

    def evaluate(self, tag, work_dir, feature_types, best_models, analysis_type="dynamic"):
        """
        Evaluate the performance of classifier(s) on our evaluation dataset.
        """
        # Updating whether to handle hybrid models
        DATASET_MAP = self.setup_eval_dataset(analysis_type)
        
        if DATASET_MAP is None:
            print("Something went wrong while setting up the evaluation dataset, Please check if all the paths are valid and all the files referred to exist in the classifier/data directory.")
            return

        print("\n" + "="*60)
        print("Running evaluation on classifier(s)")
        print("="*60)

        res = defaultdict(list)

        cluster_dataset = DATASET_MAP[tag] # Eval dataset for a particular cluster/task tag
        for dataset_type in cluster_dataset:

            ben_wild_data = cluster_dataset[dataset_type]
            syscall_data = self.extract_features(ben_wild_data, feature_types=feature_types) # Change this to adjust based on the model being tested on

            print(f"Running evaluation on {dataset_type}")
            for best_model in best_models:
                # df = pd.DataFrame(columns=["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "roc_auc"])
                print("Evaluating on model:", best_model)
                p = Path(best_model)
                dir_name = "/".join(p.parent.parts[-2:])
                clf_name = os.path.join(dir_name, os.path.basename(best_model))

                metadata_dir = os.path.join(work_dir, "metadata", "model", tag, clf_name)
                # eval_log = os.path.join(metadata_dir, "evaluation_metrics_log.csv")
                anomaly_log = os.path.join(metadata_dir, f"anomalous_models_{dataset_type}.csv")

                os.makedirs(metadata_dir, exist_ok=True)

                self.load_model(path=best_model)
                
                predictions, dscores = self.predict(syscall_data)
                print("Predictions:", predictions)

                anomalous_df = pd.DataFrame(columns=["name", "filename", "decision_score"]) # Set of models detected as anomaly
                # Save all the models which are detected as anomalies
                for i, (idx, row) in enumerate(ben_wild_data.iterrows()):
                    model_name = row["name"]
                    filename = ""
                    if "filename" in row:
                        filename = row["filename"]
                    if predictions[i] == -1:
                        try:
                            anomalous_df.loc[len(anomalous_df)] = {
                                "name": model_name,
                                "filename": filename,
                                "decision_score": dscores[i]
                            }
                            print(f"Anomaly detected in model: {model_name}")
                        except Exception as e:
                            print(f"Error processing model {model_name}: {e}")
                            continue
                    else:
                        print(f"No anomaly detected in model: {model_name}")

                res[dataset_type].append((clf_name, len(anomalous_df)))
                anomalous_df.to_csv(anomaly_log, index=False)

                # explain_model_with_shap(clf, ben_wild_data, num_samples_to_explain=25, plot_save_dir=metadata_dir, feature_types=feature_types)

        for dataset_type in cluster_dataset:
            print(f"Number of detections for {dataset_type}")
            for clf_name, num_detections in res[dataset_type]:
                print(f"{clf_name} : {num_detections}")
            print("=" * 60)

def get_model_class(model_type):
    """
    Get the respective class of the model type inputted.
    """
    if model_type == "ocsvm":
        return OneClassSVM
    elif model_type == "sgdsvm":
        return SGDOneClassSVM
    elif model_type == "isoforest":
        return IsolationForest
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_mal_data_dist(num_benign, num_malhug, test_val_split=0.2):
    """
    Getting the malicious data distribution for validation and test set
    """
    
    # Since test and validation set is chosen to be 0.1
    num_slots = num_benign * test_val_split

    remaining_slots = num_slots - num_malhug
    return int(remaining_slots)

def explain_model_with_shap(detector_instance, test_data, num_samples_to_explain=10, plot_save_dir=None):
    """
    Explain the model's predictions using the SHAP library.
    """
    print("\n=== Generating SHAP Explanations ===")

    test_syscall_dicts = detector_instance.extract_features(test_data, feature_types=args.feature_types)
    X_counts = detector_instance.vectorizer.transform(test_syscall_dicts)
    
    feature_names = detector_instance.vectorizer.get_feature_names_out()
    frequency_indices = [i for i, name in enumerate(feature_names) if name.startswith('frequency_') or name.startswith('seq')]
    
    X_scaled = X_counts.copy()
    if frequency_indices:
        X_scaled[:, frequency_indices] = detector_instance.scaler.transform(X_counts[:, frequency_indices])

    data_to_explain = X_scaled[:num_samples_to_explain] # Select the data to be explained later
    
    background_data = shap.sample(X_scaled, 100) # Random sampling of 100 rows for setting up baseline data for SHAP

    if detector_instance.classifier_class in [SGDOneClassSVM, OneClassSVM]:
        explainer = shap.KernelExplainer(detector_instance.model.decision_function, background_data)
        shap_values = explainer.shap_values(data_to_explain)
    elif detector_instance.classifier_class == IsolationForest:
        explainer = shap.TreeExplainer(detector_instance.model)
        shap_values = explainer.shap_values(data_to_explain)
    else:
        raise ValueError("Unsupported classifier class. Only SGDOneClassSVM and OneClassSVM are supported.")

    out_dir = os.path.join(plot_save_dir, "shap_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    # force_plot = shap.force_plot(explainer.expected_value, shap_values[0], data_to_explain[0], feature_names=feature_names)
    # shap.save_html(os.path.join(out_dir, "force_sample_1.html"), force_plot)

    try:
        plt.figure(figsize=(10, 2.5))
        shap.force_plot(explainer.expected_value, shap_values[0], data_to_explain[0], feature_names=feature_names, matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "force_sample_1.png"), dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.waterfall_plot(shap.Explanation(values=shap_values[1], 
                            base_values=explainer.expected_value, 
                            data=data_to_explain[1].tolist(), # Use original scaled data
                            feature_names=feature_names),show=False)

        # shap.save_html(os.path.join(out_dir, "force_sample_1.html"), force_plot, show=True)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "waterfall_sample_0.png"), dpi=200, bbox_inches="tight")
        plt.close() 
    except Exception as e:
        print(f"Static force plot save skipped: {e}")

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, data_to_explain, feature_names=feature_names, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots to: {out_dir}")
    
def validate_feature_types(feature_types):
    print("Validating feature types")
    for feat in feature_types:
        if feat not in ALLOWED_FEATURE_TYPES:
            print(f"Error: Invalid feature type: {feat}")
            sys.exit(1)
    print("All features are valid!")

if __name__ == "__main__":
    ALLOWED_FEATURE_TYPES = {"presence", "frequency", "sequence_presence", "sequence_frequency", "gen_presence", "gen_frequency", "gen_seq_presence", "gen_seq_frequency", "proc_seq_presence", "proc_seq_frequency"}

    # TODO: fix all args.hybrid and args.static references
    parser = argparse.ArgumentParser(description="Train and classify models using One Class SVM")
    parser.add_argument("--work-dir", default=os.getcwd(), help="Working directory")
    parser.add_argument("--tag", required=True, help="Task tag of the classifier being trained.")
    parser.add_argument("--evaluate", action="store_true", help="Whether or not the model should be evaluated.")
    parser.add_argument("--tune-hyperparam", action="store_true", help="Whether or not hyperparameters should be tuned during training")
    parser.add_argument("--num-benign-samples", type=int, required=False, help="Number of benign samples to use for training or evaluation")
    parser.add_argument("--feature-types", nargs='+', required=True, default=["presence", "frequency"], help="Types of features to extract from syscall data (presence, frequency, sequence_presence, sequence_frequency, proc_seq_presence, proc_seq_frequency, gen_presence, gen_frequency, gen_seq_presence, gen_seq_frequency)")
    parser.add_argument("--analysis-type", help="Whether to train/evaluate dynamic/static/hybrid models", choices=["dynamic", "static", "hybrid"])
    parser.add_argument("--model-type", choices=["ocsvm", "sgdsvm", "isoforest"], default="sgdsvm", help="Type of classifier model to use (default: sgdsvm)")
    parser.add_argument("--mode", help="Custom suffix to add onto model folder and file names.", default="")
    parser.add_argument("--best-models", help="The paths of the best models to evaluate on.", nargs='+')
    args = parser.parse_args()

    CLASSIFIER_DIR = os.path.join(args.work_dir, "classifier")
    model_dir = os.path.join(CLASSIFIER_DIR, "models", args.tag)
    
    validate_feature_types(args.feature_types)
    detector = SyscallAnomalyDetector(model_path=model_dir, hybrid=(args.analysis_type == "static" or args.analysis_type == "hybrid"), classifier_class=get_model_class(args.model_type), classifier_params={"nu": 0.05, "gamma": "scale", "kernel": "rbf", "batch_size": 32})  

    if args.tune_hyperparam:
        BENIGN_SYSCALL_DATA, BENIGN_OPCODE_DATA, MALICIOUS_SYSCALL_DATA, MALICIOUS_OPCODE_DATA, DYNAMIC_INJECTED_DATA, STATIC_INJECTED_DATA = detector.setup_train_dataset()[args.tag]
        
        if detector.hybrid and (not BENIGN_OPCODE_DATA or not MALICIOUS_OPCODE_DATA or not STATIC_INJECTED_DATA):
            print(f"No static data provided for static/hybrid mode in {args.tag}. Please choose 'text-generation' if you would like to train a hybrid model.")
            sys.exit(1)

        if not os.path.exists(BENIGN_SYSCALL_DATA) or (BENIGN_OPCODE_DATA and not os.path.exists(BENIGN_OPCODE_DATA)) or (MALICIOUS_OPCODE_DATA and not os.path.exists(MALICIOUS_OPCODE_DATA)) or not os.path.exists(MALICIOUS_SYSCALL_DATA):
            print(f"Benign data or malicious data file not found: {BENIGN_SYSCALL_DATA}, {MALICIOUS_OPCODE_DATA}, {BENIGN_OPCODE_DATA}, {MALICIOUS_SYSCALL_DATA}")
            print("Please ensure the file exists and contains syscall count data.")
            sys.exit(1)

        # Setting up the training datasets
        if args.analysis_type == "static":
            # static models
            print("Training static model")
            ben_data = pd.read_csv(BENIGN_OPCODE_DATA).head(args.num_benign_samples)
            mal_data = pd.read_csv(MALICIOUS_OPCODE_DATA)
            num_inj = get_mal_data_dist(args.num_benign_samples, len(mal_data))
            stat_inj_data = pd.read_csv(STATIC_INJECTED_DATA).sample(num_inj, random_state=42)
            mal_data = pd.concat([mal_data, stat_inj_data], axis=0)
        else:
            # dynamic/hybrid models
            # Loading the training and evaluation data
            hybrid = args.analysis_type == "hybrid"
            ben_data = pd.read_parquet(BENIGN_SYSCALL_DATA, engine='pyarrow').head(args.num_benign_samples)
            if hybrid:
                ben_opcode_data = pd.read_csv(BENIGN_OPCODE_DATA)
                ben_data = pd.merge(ben_data, ben_opcode_data, on=["name"])

            ben_data = ben_data.head(args.num_benign_samples)  # Limit to the specified number of benign samples
            
            mal_data = pd.read_parquet(MALICIOUS_SYSCALL_DATA, engine='pyarrow')
            num_inj = get_mal_data_dist(args.num_benign_samples, len(mal_data))
            if hybrid:
                mal_opcode_data = pd.read_csv(MALICIOUS_OPCODE_DATA)
                stat_inj_data = pd.read_csv(STATIC_INJECTED_DATA)
                dyn_inj_data = pd.read_parquet(DYNAMIC_INJECTED_DATA, engine='pyarrow')
                inj_data = pd.merge(dyn_inj_data, stat_inj_data, on=['name', 'filename']).sample(num_inj, random_state=42)
                mal_data = pd.merge(mal_data, mal_opcode_data, on=["name", "filename"])
                mal_data = pd.concat([mal_data, inj_data], axis=0)
            else:
                dyn_inj_data = pd.read_parquet(DYNAMIC_INJECTED_DATA, engine='pyarrow').sample(num_inj, random_state=42)
                mal_data = pd.concat([mal_data, dyn_inj_data], axis=0)

        print(f"Benign data of size: {len(ben_data)} and malicious dataset of size: {len(mal_data)} have been loaded.")
        print(f"Features for benign: {len(ben_data.columns)}, Features for malicious: {len(mal_data.columns)}")

        X_temp, X_test = train_test_split(ben_data, test_size=0.1, random_state=42)
        X_train, X_val = train_test_split(X_temp, test_size=1/9, random_state=42)
        


        # Perform k-fold cross-validation on benign data
        print("\n=== Performing K-Fold Cross-Validation ===")
        param_grid = {
            "ocsvm" :{
                'kernel': ["rbf", "linear", "sigmoid"],
                'gamma': ["scale", "auto", 0.01, 0.1, 1],
                'nu': [round(val, 2) for val in np.linspace(0.01, 0.5, 15).tolist()]
            },
            "sgdsvm": {
                'kernel': ["rbf", "linear", "poly", "sigmoid"],
                'gamma': [0.01, 0.1, 1], # Nystroem doesn't accept string values like auto and scale
                'nu': [round(val, 2) for val in np.linspace(0.01, 0.5, 10).tolist()],
                'batch_size': [32, 64, 128, 1000] # one with the entire dataset as the batch size
            },
            "isoforest": {
                'n_estimators': np.arange(50, 151, 25), # number of trees in the forest
                'max_samples': [128, 256, 512, 1000],
                'contamination': [0.001, 0.01, 0.1, "auto"], # expected proportion of outliers in the data -> calculated decision
                'random_state': [42]
            }
        }

        grid = ParameterGrid(param_grid[args.model_type])
        best_score = 0 # f1 score across all folds
        best_params = {}
        best_results = None
        best_model_data = None
        best_fold_dir = None
        all_best_models = [] # list of all models which have the best f1 score

        for params in grid:
            results, fold_model, fold_dir = detector.perform_k_fold(ben_data, mal_data, mode=args.mode, feature_types=args.feature_types, params=params, k=5)

            f1_score = results['f1'].mean()
            if f1_score > best_score:
                best_score = f1_score
                best_params = params
                best_results = results
                best_model_data = fold_model
                best_fold_dir = fold_dir
                all_best_models = [(best_score, best_params, best_results, best_model_data, best_fold_dir)]
            elif f1_score == best_score:
                all_best_models.append((best_score, params, results, fold_model, fold_dir))

        print(f"There are {len(all_best_models)} present in this.")
        for best_score, best_params, best_results, best_model_data, best_fold_dir in all_best_models:
            print(f"\nBest Hyperparameters: {best_params}")
            print(f"Best F1 Score: {best_score:.4f}")

            # Save the best model and results
            hybrid_suffix = ("_hybrid" if detector.hybrid else "") + f"_{args.mode}" if args.mode != "" else ""
            feature_type_string = "_".join(args.feature_types)
            params_dir = "_".join([str(item) for kv in best_params.items() for item in kv])  
            model_name_dir = os.path.join(f"{args.num_benign_samples}_benign_data_{feature_type_string}{hybrid_suffix}_best", detector.classifier_class.__name__, f"params-{params_dir}")
            best_model_dir = os.path.join(model_dir, model_name_dir)
            os.makedirs(best_model_dir, exist_ok=True)
            
            # Save the best model files
            joblib.dump(best_model_data['model'], os.path.join(best_model_dir, "oneclass_svm_model.pkl"))
            joblib.dump(best_model_data['vectorizer'], os.path.join(best_model_dir, "vectorizer.pkl"))
            joblib.dump(best_model_data['scaler'], os.path.join(best_model_dir, "scaler.pkl"))
            
            # Save the best model metrics
            best_results.to_csv(os.path.join(best_model_dir, f"k_fold_results_hyperparams.csv"), index=False)
            with open(os.path.join(best_model_dir, "best_hyperparams.txt"), "w") as f:
                f.write(str(best_params))

            print(f"Best Average metrics across all folds for hyperparam tuning:")
            print(f"Average accuracy: {best_results['accuracy'].mean():.4f}")
            print(f"Average precision: {best_results['precision'].mean():.4f}")
            print(f"Average recall: {best_results['recall'].mean():.4f}")
            print(f"Average F1 Score: {best_results['f1'].mean():.4f}")
            print(f"Average ROC AUC: {best_results['roc_auc'].mean():.4f}")
            
            # Evaluate best model on unseen test data
            print("\n=== Evaluating Best Model on Test Set ===")
            
            detector.model = best_model_data['model']
            detector.vectorizer = best_model_data['vectorizer']
            detector.scaler = best_model_data['scaler']
            
            test_data = best_model_data['test_data']
            test_labels = best_model_data['test_labels']

            best_fold_fp_samples = best_model_data['fp_samples']

            print(f"False positives in best fold: {len(best_fold_fp_samples)}")
            if not best_fold_fp_samples.empty:
                print("False positive samples from the best fold:")
                print(best_fold_fp_samples.head(10))
            
            _, tp, tn, fp, fn, accuracy, precision, recall, f1, roc_auc, fp_rows = detector.analyze_samples(
                test_data, test_labels, feature_types=args.feature_types
            )
            
            print(f"Test Set Metrics:")
            print(f"Total test samples: {len(test_data)}")
            print(f"True Positives: {tp}, True Negatives: {tn}")
            print(f"False Positives: {fp}, False Negatives: {fn}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # Save test metrics
            test_results = pd.DataFrame([{
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'roc_auc': roc_auc,
                'total_samples': len(test_data)
            }])
            test_results.to_csv(os.path.join(best_model_dir, "test_results.csv"), index=False)

            print("\nFalse Positive Samples:")
            print(f"Total False Positive Samples: {len(fp_rows)}")
            if not fp_rows.empty:
                print(fp_rows.head(10))

            print("\nMore information on the test set:")
            print(f"Test set starts from {best_model_data['test_data_start_idx']}")
            print(f"Test set ends at {best_model_data['test_data_end_idx']}")
            # Explaining the best model 
            explain_model_with_shap(detector, test_data, num_samples_to_explain=len(test_data), plot_save_dir=best_model_dir)

            # Save the test set(optional)
            print("Saving the test dataset")
            test_set_dir = f"classifier/data/{model_name_dir}"
            os.makedirs(test_set_dir, exist_ok=True)
            mask = test_labels == 1
            benign_df = test_data.loc[mask].reset_index(drop=True)
            benign_df.to_csv(os.path.join(test_set_dir, "HF_ben_test_set_text-generation.csv"), index=False)

    if args.evaluate:
        if not args.tag:
            print("Please provide the tag (--tag) for the models to evaluate on")
            sys.exit(1)

        if not args.work_dir:
            print("Please provide the working directory (--work-dir)")
            sys.exit(1)

        if not args.feature_types:
            print("Please provide the feature types (--feature-types) for the model to use to evaluate.")
            sys.exit(1)

        if not args.best_models:
            print("Please provide the paths of the best models (--best-modes) to evaluate on.")
            sys.exit(1)

        if not args.analysis_type:
            print("Please provide the analysis type (--analysis-type) to select the evaluation dataset.")
            sys.exit(1)

        detector.evaluate(args.tag, args.work_dir, args.feature_types, args.best_models, args.analysis_type)