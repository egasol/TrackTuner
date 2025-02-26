import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from time import time


COLORS = cycle(['r', 'g', 'b', 'y', 'c', 'm', 'k'])


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_tracks(ax, json_data):
    color_map = {}
    for frame, frame_data in json_data.items():
        for track in frame_data['tracks']:
            track_id = track['id']
            if track_id not in color_map:
                color_map[track_id] = next(COLORS)
            ax.scatter(track['x'], track['y'], track['z'], label=f"ID {track_id}", color=color_map[track_id])

if __name__ == "__main__":
    json_files = ['annotations.json', 'detections.json', 'tracked_objects.json']

    t0 = time()

    fig = plt.figure(figsize=(18, 6))
    num_files = len(json_files)

    for i, json_file in enumerate(json_files):
        ax = fig.add_subplot(1, num_files, i+1, projection='3d')
        json_data = load_json_file(json_file)
        plot_tracks(ax, json_data)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Data from {json_file}')

    plt.tight_layout()
    plt.savefig('tracked_objects.png', dpi=50)

    print("Creating plots took", time() - t0, "seconds")
