import time
from utils.utils import get_cpu_time, get_model_metadata

class MetadataCollector:
    def __init__(self, time_log_path=None, anomaly_log_path=None, wild_mal_log_path=None, wild_run=False, retrieve_mal_models=False, injected_models=False):
        self.wild_run = wild_run
        self.retrieve_mal_models = retrieve_mal_models
        self.injected_models = injected_models

        self.time_log_path = time_log_path
        self.anomaly_log_path = anomaly_log_path
        self.wild_mal_log = wild_mal_log_path

        if time_log_path:
            self.time_taken_df = get_model_metadata(time_log_path, ["batch", "download_wall", "analysis_wall", "classification_wall", "upload_wall", "download_cpu", "analysis_cpu", "classification_cpu", "upload_cpu", "batch_size"])
        if anomaly_log_path:
            self.anomalous_df = get_model_metadata(anomaly_log_path, ["name", "likes", "downloads", "last_modified", "decision_score"])
        if wild_mal_log_path and retrieve_mal_models:
            self.wild_mal_models_df = get_model_metadata(self.wild_mal_log, ["name", "likes", "downloads", "last_updated","protectAiScan", "avScan", "pickleImportScan", "jFrogScan", "clf_prediction", "clf_decision_score"]) # Set of models detected as unsafe by the hugging face api
        
        self._timers = {}
        self.times = {}

    def start_timer(self, stage):
        if not self.wild_run: return
        self._timers[stage] = (time.time(), get_cpu_time())

    def stop_timer(self, stage):
        if not self.wild_run or stage not in self._timers: return
        start_wall, start_cpu = self._timers.pop(stage)
        self.times[f"{stage}_wall"] = time.time() - start_wall
        self.times[f"{stage}_cpu"] = get_cpu_time() - start_cpu

    def log_anomaly(self, anomaly_data):
        if not self.wild_run and not self.injected_models: return
        self.anomalous_df.loc[len(self.anomalous_df)] = anomaly_data

    def log_batch_times(self, batch_info):
        if not self.wild_run: return
        # Combine timing data with other batch info
        full_batch_data = {**self.times, **batch_info}
        self.time_taken_df.loc[len(self.time_taken_df)] = full_batch_data
        self.times = {} # Reset for next batch

    def save(self):
        if self.wild_run:
            if self.time_log_path:
                self.time_taken_df.to_csv(self.time_log_path, index=False)
            if self.anomaly_log_path:
                self.anomalous_df.to_csv(self.anomaly_log_path, index=False)
        
        if self.injected_models:
            if self.anomaly_log_path:
                self.anomalous_df.to_csv(self.anomaly_log_path, index=False)

        if self.retrieve_mal_models:
            if self.wild_mal_log:
                self.wild_mal_models_df.to_csv(self.wild_mal_log, index=False)
        print("Metadata saved.")