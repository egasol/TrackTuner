import json
import math
import random
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from datatypes.reference import Reference, ReferenceTrack
from datatypes.detection import Detection
from utilities import get_data_path, save_json


class TrackGenerator:
    def __init__(
        self,
        num_frames: int,
        num_tracks: 5,
        position_randomization: float = 0.1,
        delete_probability: float = 0.1,
        add_probability: float = 0.1,
    ):
        self.num_frames = num_frames
        self.num_tracks = num_tracks
        self.position_randomization = position_randomization
        self.delete_probability = delete_probability
        self.add_probability = add_probability

        self.min_track_length = 20
        self.annotations = self.generate_annotations()
        self.min_max_ranges = self.get_min_max_ranges()

    def _generate_track_ranges(self) -> Dict[int, Tuple[int, int]]:
        track_ranges = {}
        for i in range(self.num_tracks):
            start_frame = random.randint(
                1, self.num_frames - (self.min_track_length + 1)
            )
            end_frame = random.randint(
                start_frame + self.min_track_length, self.num_frames
            )
            track_ranges[i] = (start_frame, end_frame)
        return track_ranges

    def generate_annotations(self) -> Dict[int, List[Reference]]:
        track_ranges = self._generate_track_ranges()
        annotations = {frame: [] for frame in range(1, self.num_frames + 1)}

        for track_id, (start_frame, end_frame) in track_ranges.items():
            initial_x = random.uniform(-10, 10)
            initial_y = random.uniform(-10, 10)
            initial_z = random.uniform(-10, 10)
            track = ReferenceTrack(
                track_id, initial_x, initial_y, initial_z, start_frame, end_frame
            )
            track_positions = track.generate()

            for frame, reference in track_positions.items():
                annotations[frame].append(reference)

        return annotations

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

                new_track = Detection(x=track.x, y=track.y, z=track.z)
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

    def save_data(self, annotations_path: Path, detections_path: Path) -> None:
        detections = self.modify_tracks()

        save_json(
            annotations_path,
            {
                frame: {"tracks": [ref.to_dict() for ref in ref_list]}
                for frame, ref_list in self.annotations.items()
            },
        )
        save_json(
            detections_path,
            {
                frame: {"tracks": [det.to_dict() for det in det_list]}
                for frame, det_list in detections.items()
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generation of syntethic references and detections."
    )

    parser.add_argument(
        "--output-references",
        type=Path,
        help="Path to save references.",
    )
    parser.add_argument(
        "--output-detections",
        type=Path,
        help="Path to save detections.",
    )
    parser.add_argument("--num-frames", type=int, help="Number of frames.")
    parser.add_argument(
        "--num-tracks", type=int, help="Number of reference tracks create."
    )
    parser.add_argument(
        "--position-randomize",
        type=float,
        default=0.125,
        help="Randomization of position for detections. (default: %(default)s)",
    )
    parser.add_argument(
        "--delete-probability",
        type=float,
        default=0.18,
        help="Delete probability of detections. (FNr) (default: %(default)s)",
    )
    parser.add_argument(
        "--add-probability",
        type=float,
        default=1.42,
        help="Add probability of false detections. (framewise FPr) (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    track_generator = TrackGenerator(
        num_frames=args.num_frames,
        num_tracks=args.num_tracks,
        position_randomization=args.position_randomize,
        delete_probability=args.delete_probability,
        add_probability=args.add_probability,
    )

    track_generator.save_data(args.output)


if __name__ == "__main__":
    main()
