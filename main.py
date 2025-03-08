import optuna
import random
from pathlib import Path
from typing import List

from optimizer import Optimizer
from annotator import TrackGenerator
from tracker import TrackSettings, Tracker, run_tracker_with_parameters
from visualizer import Visualizer, VisualizerInput
from utilities import load_json, save_json, get_data_path, get_media_path


def _create_filelist(prefix: str, nr_files: int) -> List[str]:
    return [f"{prefix}_{nr}" for nr in range(nr_files)]


def _generate_input_data(
    references_dir: Path, detections_dir: Path, filelist: List[str], seed: int
) -> None:
    random.seed(seed)

    for file in filelist:
        references_path = references_dir / f"{file}.json"
        detections_path = detections_dir / f"{file}.json"

        track_generator = TrackGenerator(
            num_frames=100,
            num_tracks=3,
            position_randomization=0.05,
            delete_probability=0.14,
            add_probability=4.82,
        )
        track_generator.save_data(references_path, detections_path)


def run(n_files: int, n_trials: int) -> None:
    filelist = _create_filelist("clip", n_files)

    references_dir = get_data_path() / "references"
    detections_dir = get_data_path() / "detections"
    tracked_dir = get_data_path() / "tracked"
    parameters_path = get_data_path() / "parameters.json"

    # Create input data
    _generate_input_data(references_dir, detections_dir, filelist, seed=42)

    # Optimize tracker parameters
    optimizer = Optimizer(references_dir, detections_dir, filelist)
    parameters = optimizer.optimize(n_trials=n_trials)
    save_json(parameters_path, parameters)


if __name__ == "__main__":
    run(n_files=5, n_trials=10)
