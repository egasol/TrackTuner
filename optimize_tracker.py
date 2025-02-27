import json
import numpy as np
import optuna
from pathlib import Path
from optuna.visualization import plot_optimization_history, plot_param_importances
from typing import Any, Dict

from tracker import Tracker, TrackSettings, run_tracker_with_parameters
from evaluate_tracker import process_data, Statistics
import utilities


def evaluate_tracker_performance(
    annotations: Dict[str, Any], tracks: Dict[str, Any]
) -> float:
    stats = process_data(annotations, tracks)
    return stats.get_performance_metric()


def objective(trial: optuna.trial.Trial) -> float:
    tracker_settings = TrackSettings(
        measurement_noise=trial.suggest_float("measurement_noise", 0.1, 5.0),
        process_noise=trial.suggest_float("process_noise", 0.0001, 0.1),
        covariance=trial.suggest_float("covariance", 1.0, 20.0),
        distance_threshold=trial.suggest_float("distance_threshold", 2.0, 10.0),
        max_age=trial.suggest_int("max_age", 1, 10),
        min_hits=trial.suggest_int("min_hits", 1, 10),
    )

    detections = utilities.load_json(utilities.get_data_path() / "detections.json")
    annotations = utilities.load_json(utilities.get_data_path() / "annotations.json")

    tracks = run_tracker_with_parameters(tracker_settings, detections)

    return evaluate_tracker_performance(annotations, tracks)


def main() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    parameters_path = utilities.get_data_path() / "parameters.json"

    utilities.save_json(parameters_path, study.best_params)

    plot_optimization_history(study).write_image("media/optimization_history.png")
    plot_param_importances(study).write_image("media/plot_param_importances.png")


if __name__ == "__main__":
    main()
