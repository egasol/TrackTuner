import json
import numpy as np
import optuna
import statistics
from pathlib import Path
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict, List, Tuple

from tracker import Tracker, TrackSettings, run_tracker_with_parameters
from evaluator import process_data, Statistics
from utilities import load_json, save_json


class Optimizer:
    def __init__(
        self,
        references_dir: Path,
        detections_dir: Path,
        filelist: List[str],
    ):
        self.references_dir = references_dir
        self.detections_dir = detections_dir
        self.filelist = filelist
        self.study = optuna.create_study(direction="minimize")
        self.input_data: List[Tuple] = []

    def _load_data(self):
        for file in self.filelist:
            ref_path = self.references_dir / f"{file}.json"
            det_path = self.detections_dir / f"{file}.json"

            if ref_path.exists() and det_path.exists():
                references = load_json(ref_path)
                detections = load_json(det_path)

                self.input_data.append((references, detections))

        print(f"Loaded data for {len(self.input_data)} files.")

    def _evaluator_performance(
        self, annotations: Dict[str, Any], tracks: Dict[str, Any]
    ) -> float:
        stats = process_data(annotations, tracks)
        return stats.get_performance_metric()

    def objective(self, trial: optuna.trial.Trial) -> float:
        tracker_settings = TrackSettings(
            measurement_noise=trial.suggest_float("measurement_noise", 0.001, 10.0),
            process_noise=trial.suggest_float("process_noise", 0.0001, 0.1),
            covariance=trial.suggest_float("covariance", 0.001, 20.0),
            distance_threshold=trial.suggest_float("distance_threshold", 0.01, 20.0),
            max_age=trial.suggest_int("max_age", 1, 10),
            min_hits=trial.suggest_int("min_hits", 1, 10),
            max_consecutive_misses=trial.suggest_int("max_consecutive_misses", 1, 10),
        )

        performance = [
            self._evaluator_performance(
                references, run_tracker_with_parameters(tracker_settings, detections)
            )
            for references, detections in self.input_data
        ]

        return statistics.mean(performance)

    def optimize(self, n_trials: int) -> Dict:
        self._load_data()
        self.study.optimize(self.objective, n_trials=n_trials)

        return self.study.best_params
