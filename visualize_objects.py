import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from time import time
from pathlib import Path
from typing import Dict, Any
import utilities

COLORS = cycle(["r", "g", "b", "y", "c", "m", "k"])
COLOR_DEFAULT = "b"


class VisualizerInput:
    def __init__(self, filepath: Path, ignore_id: bool = False):
        self.filepath = filepath
        self.ignore_id = ignore_id


class Visualizer:
    def __init__(self, input_files: list[VisualizerInput]):
        self.input_files = input_files
        self.color_map = {}

    def load_data(self, filepath: Path) -> Dict[str, Any]:
        return utilities.load_json(filepath)

    def plot_tracks(
        self, ax: Axes3D, json_data: Dict[str, Any], ignore_id: bool
    ) -> None:
        for frame, frame_data in json_data.items():
            for track in frame_data["tracks"]:
                if not ignore_id:
                    track_id = track["id"]
                    if track_id not in self.color_map:
                        self.color_map[track_id] = next(COLORS)
                    color = self.color_map[track_id]
                else:
                    color = "b"
                ax.scatter(track["x"], track["y"], track["z"], color=color)

    def visualize(self, output: Path) -> None:
        t0 = time()
        fig = plt.figure(figsize=(18, 6))
        num_files = len(self.input_files)

        for i, input_settings in enumerate(self.input_files):
            ax = fig.add_subplot(1, num_files, i + 1, projection="3d")
            filepath = input_settings.filepath
            ignore_id = input_settings.ignore_id

            json_data = self.load_data(filepath)
            self.plot_tracks(ax, json_data, ignore_id)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(filepath.stem)

        plt.tight_layout()
        plt.savefig(output, dpi=50)
        print("Summarizing plots to", output, f"({time() - t0:.2f}s)")


if __name__ == "__main__":
    input_files = [
        VisualizerInput(utilities.get_data_path() / "annotations.json"),
        VisualizerInput(utilities.get_data_path() / "detections.json", ignore_id=True),
        VisualizerInput(utilities.get_data_path() / "tracked.json"),
    ]

    visualizer = Visualizer(input_files)
    visualizer.visualize(utilities.get_media_path() / "comparison.png")
