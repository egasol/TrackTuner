import json
import math
import random
import numpy as np
from typing import Dict, List, Any

import utilities


def generate_object_positions(frame: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": 0,
            "x": 10 * math.sin(0.1 * frame),
            "y": 15 * math.cos(0.1 * frame),
            "z": 20 + 0.1 * frame,
        },
        {
            "id": 1,
            "x": -10 * math.cos(0.1 * frame),
            "y": -15 * math.sin(0.1 * frame),
            "z": 10 * math.sin(0.1 * frame),
        },
        {
            "id": 2,
            "x": 10 * math.cos(0.1 * frame),
            "y": 15 - 0.5 * frame,
            "z": -20 * math.sin(0.1 * frame),
        },
    ]


def generate_annotations(num_frames: int) -> Dict[int, Dict[str, Any]]:
    frames = {}
    for frame in range(1, num_frames + 1):
        frames[frame] = {"tracks": generate_object_positions(frame)}
    return frames


def modify_tracks(
    json_data: Dict[int, Dict[str, Any]],
    position_randomization: float = 0.1,
    delete_probability: float = 0.1,
    add_probability: float = 0.1,
) -> Dict[int, Dict[str, Any]]:
    new_data = {}

    for frame, frame_data in json_data.items():
        new_tracks = []
        for track in frame_data["tracks"]:
            if random.random() < delete_probability:
                continue

            new_track = track.copy()
            new_track["x"] += np.random.uniform(
                -position_randomization, position_randomization
            )
            new_track["y"] += np.random.uniform(
                -position_randomization, position_randomization
            )
            new_track["z"] += np.random.uniform(
                -position_randomization, position_randomization
            )
            new_tracks.append(new_track)

        num_new_tracks = np.random.poisson(add_probability)

        for _ in range(num_new_tracks):
            new_track = {
                "id": max([t["id"] for t in new_tracks] + [-1]) + 1,
                "x": np.random.uniform(-15, 15),
                "y": np.random.uniform(-15, 15),
                "z": np.random.uniform(-15, 15),
            }
            new_tracks.append(new_track)

        new_data[frame] = {"tracks": new_tracks}

    return new_data


def main() -> None:
    num_frames = 100
    annotations_path = utilities.get_data_path() / "annotations.json"
    detections_path = utilities.get_data_path() / "detections.json"

    annotations = generate_annotations(num_frames)
    detections = modify_tracks(
        annotations,
        position_randomization=0.5,
        delete_probability=0.4,
        add_probability=1.4,
    )

    utilities.save_json(annotations_path, annotations)
    utilities.save_json(detections_path, detections)


if __name__ == "__main__":
    main()
