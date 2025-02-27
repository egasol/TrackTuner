import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from time import time
from pathlib import Path

import utilities

COLORS = cycle(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
COLOR_DEFAULT = 'b'


def plot_tracks(ax, json_data, ignore_id):
    color_map = {}
    for frame, frame_data in json_data.items():
        for track in frame_data['tracks']:
            if not ignore_id:
                track_id = track['id']
                if track_id not in color_map:
                    color_map[track_id] = next(COLORS)
                color = color_map[track_id]
            else:
                color = 'b'
            ax.scatter(track['x'], track['y'], track['z'], color=color)

if __name__ == '__main__':
    input_files = [
        {
            'filepath': utilities.get_data_path() / "annotations.json",
            'ignore_id': False
        },
        {
            'filepath': utilities.get_data_path() / "detections.json",
            'ignore_id': True
        },
        {
            'filepath': utilities.get_data_path() / "tracked.json",
            'ignore_id': False
        }
    ]

    t0 = time()

    fig = plt.figure(figsize=(18, 6))
    num_files = len(input_files)

    for i, input_settings in enumerate(input_files):
        ax = fig.add_subplot(1, num_files, i+1, projection='3d')
        filepath = input_settings['filepath']
        ignore_id = input_settings['ignore_id']

        json_data = utilities.load_json(filepath)
        plot_tracks(ax, json_data, ignore_id)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(filepath.stem)

    plt.tight_layout()
    plt.savefig(utilities.get_media_path() / 'comparison.png', dpi=50)

    print('Creating plots took', time() - t0, 'seconds')
