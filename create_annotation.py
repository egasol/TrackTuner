import json
import math
import random
import numpy as np
from typing import Dict, List, Any

import utilities

class TrackGenerator:
    def __init__(self, num_frames: int, position_randomization: float = 0.1,
                 delete_probability: float = 0.1, add_probability: float = 0.1):
        self.num_frames = num_frames
        self.position_randomization = position_randomization
        self.delete_probability = delete_probability
        self.add_probability = add_probability
        self.annotations = self.generate_annotations()

    @staticmethod
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

    def generate_annotations(self) -> Dict[int, Dict[str, Any]]:
        frames = {}
        for frame in range(1, self.num_frames + 1):
            frames[frame] = {"tracks": self.generate_object_positions(frame)}
        return frames

    def generate_false_positives(self, new_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        num_new_tracks = np.random.poisson(self.add_probability)
        for _ in range(num_new_tracks):
            new_track = {
                "id": max([t["id"] for t in new_tracks] + [-1]) + 1,
                "x": np.random.uniform(-15, 15),
                "y": np.random.uniform(-15, 15),
                "z": np.random.uniform(-15, 15),
            }
            new_tracks.append(new_track)
        return new_tracks

    def modify_tracks(self) -> Dict[int, Dict[str, Any]]:
        new_data = {}

        for frame, frame_data in self.annotations.items():
            new_tracks = []
            for track in frame_data["tracks"]:
                if random.random() < self.delete_probability:
                    continue

                new_track = track.copy()
                new_track["x"] += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_track["y"] += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_track["z"] += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_tracks.append(new_track)

            # Generate false positives
            new_tracks = self.generate_false_positives(new_tracks)

            new_data[frame] = {"tracks": new_tracks}

        return new_data

    def save_data(self) -> None:
        annotations_path = utilities.get_data_path() / "annotations.json"
        detections_path = utilities.get_data_path() / "detections.json"

        detections = self.modify_tracks()

        utilities.save_json(annotations_path, self.annotations)
        utilities.save_json(detections_path, detections)

def main() -> None:
    num_frames = 100

    track_generator = TrackGenerator(
        num_frames=num_frames,
        position_randomization=0.5,
        delete_probability=0.4,
        add_probability=1.4
    )

    track_generator.save_data()

if __name__ == "__main__":
    main()
