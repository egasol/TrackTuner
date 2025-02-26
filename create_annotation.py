import json
import math
import random
import numpy as np

def generate_object_positions(frame):
    return [
        {"id": 0, "x": 10 * math.sin(0.1 * frame), "y": 15 * math.cos(0.1 * frame), "z": 20 + 0.1 * frame},
        {"id": 1, "x": -10 * math.cos(0.1 * frame), "y": -15 * math.sin(0.1 * frame), "z": 10 * math.sin(0.1 * frame)},
        {"id": 2, "x": 10 * math.cos(0.1 * frame), "y": 15 -0.5 * frame, "z": -20 * math.sin(0.1 * frame)}
    ]

def generate_annotations(num_frames):
    frames = {}
    for frame in range(1, num_frames + 1):
        frames[frame] = {"tracks": generate_object_positions(frame)}
    return frames

def modify_tracks(json_data, position_randomization=0.1, delete_probability=0.1, add_probability=0.1):
    new_data = {}

    for frame, frame_data in json_data.items():
        new_tracks = []
        for track in frame_data['tracks']:
            # Random chance to delete a track
            if random.random() < delete_probability:
                continue

            # Randomize position
            new_track = track.copy()
            new_track['x'] += np.random.uniform(-position_randomization, position_randomization)
            new_track['y'] += np.random.uniform(-position_randomization, position_randomization)
            new_track['z'] += np.random.uniform(-position_randomization, position_randomization)
            new_tracks.append(new_track)

        # Random chance to add a new track
        if random.random() < add_probability:
            new_track = {
                'id': max([t['id'] for t in new_tracks] + [-1]) + 1, # Assign a new unique ID
                'x': np.random.uniform(-15, 15), # Random X position
                'y': np.random.uniform(-15, 15), # Random Y position
                'z': np.random.uniform(-15, 15)  # Random Z position
            }
            new_tracks.append(new_track)

        new_data[frame] = {'tracks': new_tracks}

    return new_data

def main():
    num_frames = 100
    annotations = generate_annotations(num_frames)
    
    with open('annotations.json', 'w') as json_file:
        json.dump(annotations, json_file, indent=4)
    
    detections = modify_tracks(annotations, position_randomization=0.5, delete_probability=0.2, add_probability=0.2)

    with open('detections.json', 'w') as json_file:
        json.dump(detections, json_file, indent=4)

if __name__ == "__main__":
    main()
