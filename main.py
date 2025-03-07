import optuna
import random

import annotator
import optimizer
import tracker
import utilities
from visualizer import Visualizer, VisualizerInput

if __name__ == "__main__":
    annotations_path = utilities.get_data_path() / "annotations.json"
    detections_path = utilities.get_data_path() / "detections.json"
    tracked_path = utilities.get_data_path() / "tracked.json"
    parameters_path = utilities.get_data_path() / "parameters.json"

    random.seed(42)

    # Generate annotations and detections
    track_generator = annotator.TrackGenerator(
        num_frames=100,
        num_tracks=3,
        position_randomization=0.05,
        delete_probability=0.14,
        add_probability=4.82,
    )
    track_generator.save_data()

    # Optimize parameters
    study = optuna.create_study(direction="minimize")
    study.optimize(optimizer.objective, n_trials=200)
    utilities.save_json(parameters_path, study.best_params)

    # Run tracker on optimized parameters
    detections = utilities.load_json(detections_path)
    tracker_settings = tracker.TrackSettings(
        measurement_noise=study.best_params["measurement_noise"],
        process_noise=study.best_params["process_noise"],
        covariance=study.best_params["covariance"],
        distance_threshold=study.best_params["distance_threshold"],
        max_age=study.best_params["max_age"],
        min_hits=study.best_params["min_hits"],
        max_consecutive_misses=study.best_params["max_consecutive_misses"],
    )
    output_data = tracker.run_tracker_with_parameters(tracker_settings, detections)
    utilities.save_json(tracked_path, output_data)

    # Visualize results
    visualizer = Visualizer(
        [
            VisualizerInput(annotations_path),
            VisualizerInput(detections_path, ignore_id=True),
            VisualizerInput(tracked_path),
        ]
    )
    visualizer.visualize(utilities.get_media_path() / "comparison.png")
