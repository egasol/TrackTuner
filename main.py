import optuna
import random
from typing import List

import optimizer  # TODO: Change to import Optimizer class when available

from annotator import TrackGenerator
from tracker import TrackSettings, Tracker, run_tracker_with_parameters
from utilities import load_json, save_json, get_data_path, get_media_path
from visualizer import Visualizer, VisualizerInput


def run_single() -> None:
    annotations_path = get_data_path() / "annotations.json"
    detections_path = get_data_path() / "detections.json"
    tracked_path = get_data_path() / "tracked.json"
    parameters_path = get_data_path() / "parameters.json"

    random.seed(42)

    # Generate annotations and detections
    track_generator = annotator.TrackGenerator(
        num_frames=100,
        num_tracks=3,
        position_randomization=0.05,
        delete_probability=0.14,
        add_probability=4.82,
    )
    track_generator.save_data(annotations_path, detections_path)

    # Optimize parameters
    study = optuna.create_study(direction="minimize")
    study.optimize(optimizer.objective, n_trials=200)
    save_json(parameters_path, study.best_params)

    # Run tracker on optimized parameters
    detections = load_json(detections_path)
    tracker_settings = TrackSettings(
        measurement_noise=study.best_params["measurement_noise"],
        process_noise=study.best_params["process_noise"],
        covariance=study.best_params["covariance"],
        distance_threshold=study.best_params["distance_threshold"],
        max_age=study.best_params["max_age"],
        min_hits=study.best_params["min_hits"],
        max_consecutive_misses=study.best_params["max_consecutive_misses"],
    )
    output_data = run_tracker_with_parameters(tracker_settings, detections)
    save_json(tracked_path, output_data)

    # Visualize results
    visualizer = Visualizer(
        [
            VisualizerInput(annotations_path),
            VisualizerInput(detections_path, ignore_id=True),
            VisualizerInput(tracked_path),
        ]
    )
    visualizer.visualize(get_media_path() / "comparison.png")


if __name__ == "__main__":
    run_single()
