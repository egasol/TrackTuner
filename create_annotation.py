import json
import math
import random
import numpy as np
from typing import List, Tuple, Dict
from datatypes.reference import Reference
from datatypes.detection import Detection

import utilities


class TrackGenerator:
    def __init__(
        self,
        num_frames: int,
        position_randomization: float = 0.1,
        delete_probability: float = 0.1,
        add_probability: float = 0.1,
    ):
        self.num_frames = num_frames
        self.position_randomization = position_randomization
        self.delete_probability = delete_probability
        self.add_probability = add_probability
        self.annotations = self.generate_annotations()
        self.min_max_ranges = self.get_min_max_ranges()

    @staticmethod
    def generate_object_positions(frame: int) -> List[Reference]:
        return [
            Reference(
                id=0,
                x=10 * math.sin(0.1 * frame),
                y=15 * math.cos(0.1 * frame),
                z=20 + 0.1 * frame,
            ),
            Reference(
                id=1,
                x=-10 * math.cos(0.1 * frame),
                y=-15 * math.sin(0.1 * frame),
                z=10 * math.sin(0.1 * frame),
            ),
            Reference(
                id=2,
                x=10 * math.cos(0.1 * frame),
                y=15 - 0.5 * frame,
                z=-20 * math.sin(0.1 * frame),
            ),
        ]

    def generate_annotations(self) -> Dict[int, List[Reference]]:
        frames = {}
        for frame in range(1, self.num_frames + 1):
            frames[frame] = self.generate_object_positions(frame)
        return frames

    def get_min_max_ranges(self) -> Dict[str, Tuple[float, float]]:
        x_values, y_values, z_values = [], [], []
        for frame_data in self.annotations.values():
            for track in frame_data:
                x_values.append(track.x)
                y_values.append(track.y)
                z_values.append(track.z)

        return {
            "x": (min(x_values), max(x_values)),
            "y": (min(y_values), max(y_values)),
            "z": (min(z_values), max(z_values)),
        }

    def generate_false_positives(self, new_tracks: List[Detection]) -> List[Detection]:
        num_new_tracks = np.random.poisson(self.add_probability)
        x_range, y_range, z_range = (
            self.min_max_ranges["x"],
            self.min_max_ranges["y"],
            self.min_max_ranges["z"],
        )
        for _ in range(num_new_tracks):
            new_track = Detection(
                id=max([t.id for t in new_tracks] + [-1]) + 1,
                x=np.random.uniform(x_range[0], x_range[1]),
                y=np.random.uniform(y_range[0], y_range[1]),
                z=np.random.uniform(z_range[0], z_range[1]),
            )
            new_tracks.append(new_track)
        return new_tracks

    def modify_tracks(self) -> Dict[int, List[Detection]]:
        new_data = {}

        for frame, frame_data in self.annotations.items():
            new_tracks = []
            for track in frame_data:
                if random.random() < self.delete_probability:
                    continue

                new_track = Detection(id=track.id, x=track.x, y=track.y, z=track.z)
                new_track.x += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_track.y += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_track.z += np.random.uniform(
                    -self.position_randomization, self.position_randomization
                )
                new_tracks.append(new_track)

            new_tracks = self.generate_false_positives(new_tracks)

            new_data[frame] = new_tracks

        return new_data

    def save_data(self) -> None:
        annotations_path = utilities.get_data_path() / "annotations.json"
        detections_path = utilities.get_data_path() / "detections.json"

        detections = self.modify_tracks()

        utilities.save_json(
            annotations_path,
            {
                frame: {"tracks": [ref.to_dict() for ref in ref_list]}
                for frame, ref_list in self.annotations.items()
            },
        )
        utilities.save_json(
            detections_path,
            {
                frame: {"tracks": [det.to_dict() for det in det_list]}
                for frame, det_list in detections.items()
            },
        )


def main() -> None:
    track_generator = TrackGenerator(
        num_frames=100,
        position_randomization=0.5,
        delete_probability=0.4,
        add_probability=1.4,
    )

    track_generator.save_data()


if __name__ == "__main__":
    main()
