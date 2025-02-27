import json
import numpy as np
import optuna
from pathlib import Path
from tracker import Tracker, TrackSettings, load_json, run_tracker_with_parameters
from evaluate_tracker import process_data, Statistics

PARAMETER_FILE = Path("parameters.json")

def evaluate_tracker_performance(annotations, tracks):  
    stats = process_data(annotations, tracks)
    return stats.get_performance_metric()

def objective(trial):
    tracker_settings = TrackSettings(
        measurement_noise=trial.suggest_float('measurement_noise', 0.1, 5.0),
        process_noise=trial.suggest_float('process_noise', 0.0001, 0.1),
        covariance=trial.suggest_float('covariance', 1.0, 20.0),
        distance_threshold=trial.suggest_float('distance_threshold', 2.0, 10.0),
        max_age=trial.suggest_int('max_age', 1, 10),
        min_hits=trial.suggest_int('min_hits', 1, 10)
    )

    detections = load_json('detections.json')
    annotations = load_json('annotations.json')

    tracks = run_tracker_with_parameters(tracker_settings, detections)

    return evaluate_tracker_performance(annotations, tracks)

def save_parameters(study):
    with open(PARAMETER_FILE, "w") as f:
        json.dump(study.best_params, f, indent=4)

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print(f"Best Settings: {study.best_params}")
    print(f"Best Performance Metric: {study.best_value}")
    save_parameters(study)

if __name__ == "__main__":
    main()
