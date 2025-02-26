import json
import numpy as np
from enum import Enum
from filterpy.kalman import KalmanFilter

class TrackStage(Enum):
    INITIALIZED = 1
    CONFIRMED = 2

class TrackSettings:
    def __init__(self, measurement_noise, process_noise, covariance, distance_threshold, max_age, min_hits):
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.covariance = covariance
        self.distance_threshold = distance_threshold
        self.max_age = max_age
        self.min_hits = min_hits

class Track:
    def __init__(self, id, initial_position, settings):
        self.measurement_noise = settings.measurement_noise
        self.covariance = settings.covariance
        self.process_noise = settings.process_noise
        self.id = id
        self.position = initial_position
        self.kf = self.initialize_kalman_filter(initial_position)
        self.stage = TrackStage.INITIALIZED
        self.age = 0
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0

    def initialize_kalman_filter(self, initial_position):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([[1,0,0,1,0,0],
                         [0,1,0,0,1,0],
                         [0,0,1,0,0,1],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0]])
        kf.R *= self.measurement_noise # Increase measurement noise
        kf.P *= self.covariance  # Covariance matrix
        kf.Q *= self.process_noise  # Process noise
        kf.x[:3] = initial_position.reshape((3, 1))
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1

    def update(self, measurement):
        self.kf.update(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        return self.kf.x[:3].reshape((3,))

    def __repr__(self):
        return f"Track {self.id}: {self.get_state()} | Stage: {self.stage}"

class Tracker:
    # def __init__(self, distance_threshold=5.0, max_age=3, min_hits=3):
    def __init__(self, settings):
        self.tracks = []
        self.track_id = 0
        self.distance_threshold = settings.distance_threshold
        self.max_age = settings.max_age
        self.min_hits = settings.min_hits
        self.settings = settings

    def associate_detections_to_tracks(self, detections):
        assigned_tracks = []
        unassigned_tracks = list(range(len(self.tracks)))
        assigned_detections = []

        for i, track in enumerate(self.tracks):
            min_distance = float('inf')
            best_match = -1
            for j, detection in enumerate(detections):
                if j in assigned_detections:
                    continue
                distance = np.linalg.norm(track.get_state() - detection)
                if distance < min_distance and distance < self.distance_threshold:
                    min_distance = distance
                    best_match = j
            if best_match != -1:
                track.update(detections[best_match])
                assigned_tracks.append(i)
                assigned_detections.append(best_match)

        unassigned_detections = [i for i in range(len(detections)) if i not in assigned_detections]
        return assigned_tracks, unassigned_tracks, unassigned_detections

    def predict_tracks(self):
        for track in self.tracks:
            track.predict()

    def update_tracks(self, detections):
        assigned_tracks, unassigned_tracks, unassigned_detections = self.associate_detections_to_tracks(detections)

        for i in unassigned_detections:
            self.tracks.append(Track(self.track_id, detections[i], self.settings))
            self.track_id += 1

        for i in reversed(unassigned_tracks):
            self.tracks[i].time_since_update += 1
            if self.tracks[i].time_since_update > self.max_age:
                self.tracks.pop(i)

        for track in self.tracks:
            if track.hits >= self.min_hits and track.stage == TrackStage.INITIALIZED:
                track.stage = TrackStage.CONFIRMED
            if track.time_since_update > 1:
                track.hit_streak = 0

    def get_tracks(self):
        return self.tracks

    # def save_results(self, file_path):
    #     with open(file_path, 'w') as json_file:
    #         json.dump(self.results, json_file, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# def run_tracker

def run_tracker_with_parameters(tracker_settings, detections):
    tracker = Tracker(tracker_settings)
    output_data = {}

    for frame, content in detections.items():
        frame_detections = [np.array([obj["x"], obj["y"], obj["z"]]) for obj in content["tracks"]]

        tracker.predict_tracks()
        tracker.update_tracks(frame_detections)

        frame_tracks = []
        for track in tracker.get_tracks():
            if track.stage == TrackStage.CONFIRMED:
                frame_tracks.append({
                    "id": track.id,
                    "x": track.get_state()[0],
                    "y": track.get_state()[1],
                    "z": track.get_state()[2]
                })

        output_data[frame] = {"tracks": frame_tracks}
        # print(f"Frame {frame}:")
        # for track in frame_tracks:
        #     print(track)
    return output_data

def main():
    detections = load_json('detections.json')
    tracker_settings = TrackSettings(
        measurement_noise=4.235956235432715,
        process_noise=0.013857524486552263,
        covariance=16.213154333483708,
        distance_threshold=8.496566110961627,
        max_age=2,
        min_hits=1
    )
    output_data = run_tracker_with_parameters(tracker_settings, detections)

    save_to_json(output_data, 'tracked_objects.json')

if __name__ == "__main__":
    main()
