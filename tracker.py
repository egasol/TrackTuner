import json
import numpy as np
import argparse
from pathlib import Path
from enum import Enum
from filterpy.kalman import KalmanFilter
from typing import Any, Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from utilities import get_data_path, load_json, save_json


class TrackStage(Enum):
    INITIALIZED = 1
    CONFIRMED = 2


class TrackSettings:
    def __init__(
        self,
        measurement_noise: float,
        process_noise: float,
        covariance: float,
        distance_threshold: float,
        max_age: int,
        min_hits: int,
        max_consecutive_misses: int,
    ) -> None:
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.covariance = covariance
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_consecutive_misses = max_consecutive_misses


class Track:
    def __init__(
        self,
        id: int,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        initial_acceleration: np.ndarray,
        settings: TrackSettings,
    ) -> None:
        self.measurement_noise = settings.measurement_noise
        self.covariance = settings.covariance
        self.process_noise = settings.process_noise
        self.id = id
        self.kf = self.initialize_kalman_filter(
            initial_position, initial_velocity, initial_acceleration
        )
        self.stage = TrackStage.INITIALIZED
        self.age = 0
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.position_history = [initial_position]

    def initialize_kalman_filter(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        initial_acceleration: np.ndarray,
    ) -> KalmanFilter:
        kf = KalmanFilter(dim_x=9, dim_z=3)
        kf.F = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0.5, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0.5, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 0.5],
                [0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        kf.R *= self.measurement_noise
        kf.P *= self.covariance
        kf.Q *= self.process_noise
        kf.x[:3] = initial_position.reshape((3, 1))
        kf.x[3:6] = initial_velocity.reshape((3, 1))
        kf.x[6:] = initial_acceleration.reshape((3, 1))
        return kf

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        self.consecutive_misses += 1
        return self.kf.x

    def update(self, measurement: np.ndarray) -> None:
        self.kf.update(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.consecutive_misses = 0
        self.position_history.append(measurement)
        if len(self.position_history) > 5:
            self.position_history.pop(0)

    def get_state(self) -> np.ndarray:
        return self.kf.x[:3].reshape((3,))

    def get_velocity(self) -> np.ndarray:
        return self.kf.x[3:6].reshape((3,))

    def get_acceleration(self) -> np.ndarray:
        return self.kf.x[6:].reshape((3,))

    def get_smoothed_position(self) -> np.ndarray:
        return np.mean(self.position_history, axis=0)

    def __repr__(self) -> str:
        return (
            f"Track {self.id}: {self.get_state()} | "
            f"Velocity: {self.get_velocity()} | "
            f"Acceleration: {self.get_acceleration()} | "
            f"Stage: {self.stage} | "
            f"Age: {self.age} | "
            f"Hits: {self.hits} | "
            f"HitStreak: {self.hit_streak} | "
            f"TimeSinceUpdate: {self.time_since_update} | "
            f"ConsecutiveMisses: {self.consecutive_misses}"
        )


class Tracker:
    def __init__(self, settings: TrackSettings) -> None:
        self.tracks: List[Track] = []
        self.track_id = 0
        self.distance_threshold = settings.distance_threshold
        self.max_age = settings.max_age
        self.min_hits = settings.min_hits
        self.max_consecutive_misses = settings.max_consecutive_misses
        self.settings = settings

    def associate_detections_to_tracks(
        self, detections: List[np.ndarray]
    ) -> Tuple[List[int], List[int], List[int]]:
        assigned_tracks: List[int] = []
        unassigned_tracks: List[int] = list(range(len(self.tracks)))
        assigned_detections: List[int] = []

        if len(detections) == 0:
            return assigned_tracks, unassigned_tracks, list(range(len(detections)))

        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            predicted_state = track.get_state()
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(predicted_state - detection)

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        for track_index, detection_index in zip(track_indices, detection_indices):
            if cost_matrix[track_index, detection_index] < self.distance_threshold:
                self.tracks[track_index].update(detections[detection_index])
                assigned_tracks.append(track_index)
                assigned_detections.append(detection_index)

        unassigned_tracks = [
            i for i in range(len(self.tracks)) if i not in assigned_tracks
        ]
        unassigned_detections = [
            i for i in range(len(detections)) if i not in assigned_detections
        ]

        return assigned_tracks, unassigned_tracks, unassigned_detections

    def predict_tracks(self) -> None:
        for track in self.tracks:
            track.predict()

    def update_tracks(self, detections: List[np.ndarray]) -> None:
        assigned_tracks, unassigned_tracks, unassigned_detections = (
            self.associate_detections_to_tracks(detections)
        )

        for i in unassigned_detections:
            initial_velocity = np.zeros(3)
            initial_acceleration = np.zeros(3)
            self.tracks.append(
                Track(
                    self.track_id,
                    detections[i],
                    initial_velocity,
                    initial_acceleration,
                    self.settings,
                )
            )
            self.track_id += 1

        for i in reversed(unassigned_tracks):
            self.tracks[i].time_since_update += 1
            if (
                self.tracks[i].time_since_update > self.max_age
                or self.tracks[i].consecutive_misses > self.max_consecutive_misses
            ):
                self.tracks.pop(i)

        for track in self.tracks:
            if track.hits >= self.min_hits and track.stage == TrackStage.INITIALIZED:
                track.stage = TrackStage.CONFIRMED
            if track.time_since_update > 1:
                track.hit_streak = 0

    def get_tracks(self) -> List[Track]:
        return self.tracks


def run_tracker_with_parameters(
    tracker_settings: TrackSettings, detections: Dict[str, Any]
) -> Dict[str, Any]:
    tracker = Tracker(tracker_settings)
    output_data: Dict[str, Any] = {}

    for frame, content in detections.items():
        frame_detections = [
            np.array([obj["x"], obj["y"], obj["z"]]) for obj in content["tracks"]
        ]

        tracker.predict_tracks()
        tracker.update_tracks(frame_detections)

        frame_tracks = []
        for track in tracker.get_tracks():
            if track.stage == TrackStage.CONFIRMED:
                frame_tracks.append(
                    {
                        "id": track.id,
                        "x": track.get_smoothed_position()[0],
                        "y": track.get_smoothed_position()[1],
                        "z": track.get_smoothed_position()[2],
                        "vx": track.get_velocity()[0],
                        "vy": track.get_velocity()[1],
                        "vz": track.get_velocity()[2],
                        "ax": track.get_acceleration()[0],
                        "ay": track.get_acceleration()[1],
                        "az": track.get_acceleration()[2],
                    }
                )

        output_data[frame] = {"tracks": frame_tracks}
    return output_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tracker given detections and tracker parameters."
    )

    parser.add_argument(
        "--input-detections",
        type=Path,
        help="Path to detections json file.",
    )
    parser.add_argument(
        "--input-parameters",
        type=Path,
        help="Path to parameters load json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output tracks json file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detections = load_json(args.input_detections)
    parameters = load_json(args.input_parameters)
    tracker_settings = TrackSettings(
        measurement_noise=parameters["measurement_noise"],
        process_noise=parameters["process_noise"],
        covariance=parameters["covariance"],
        distance_threshold=parameters["distance_threshold"],
        max_age=parameters["max_age"],
        min_hits=parameters["min_hits"],
        max_consecutive_misses=parameters["max_consecutive_misses"],
    )
    output_data = run_tracker_with_parameters(tracker_settings, detections)

    save_json(args.output, output_data)


if __name__ == "__main__":
    main()
