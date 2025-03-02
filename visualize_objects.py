import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from time import time
from pathlib import Path
from typing import Dict, Any
import utilities


class VisualizerInput:
    def __init__(self, filepath: Path, ignore_id: bool = False):
        self.filepath = filepath
        self.ignore_id = ignore_id


class Visualizer:
    def __init__(self, input_files: list[VisualizerInput]):
        self.input_files = input_files
        self.color_map = {}
        self.alpha_min = 0.05
        self.alpha_max = 1.00
        self.colors = cycle(["r", "g", "b", "y", "c", "m", "k"])
        self.color_default = "b"

    def _load_data(self, filepath: Path) -> Dict[str, Any]:
        return utilities.load_json(filepath)

    def _plot_tracks(
        self,
        ax: Axes3D,
        json_data: Dict[str, Any],
    ) -> None:
        track_points = {}

        for frame_data in json_data.values():
            for track in frame_data["tracks"]:
                track_id = track.get("id")
                if track_id not in track_points:
                    track_points[track_id] = {"x": [], "y": [], "z": []}
                track_points[track_id]["x"].append(track["x"])
                track_points[track_id]["y"].append(track["y"])
                track_points[track_id]["z"].append(track["z"])

        for track_id, points in track_points.items():
            if track_id not in self.color_map:
                self.color_map[track_id] = next(self.colors)
            color = self.color_map[track_id]

            ax.plot(points["x"], points["y"], points["z"], color=color)

    def _plot_tracks_scatter(self, ax: Axes3D, json_data: Dict[str, Any]) -> None:
        track_data_by_id = {}

        max_frame = max(map(int, json_data.keys()))
        for frame_str, frame_data in json_data.items():
            frame = int(frame_str)
            alpha = self.alpha_min + (frame / max_frame) * (
                self.alpha_max - self.alpha_min
            )

            for track in frame_data["tracks"]:
                x, y, z = track["x"], track["y"], track["z"]

                ax.scatter(x, y, z, color=self.color_default, alpha=alpha)

    def visualize(self, output: Path) -> None:
        t0 = time()
        fig = plt.figure(figsize=(18, 6))
        num_files = len(self.input_files)

        for i, input_settings in enumerate(self.input_files):
            ax = fig.add_subplot(1, num_files, i + 1, projection="3d")

            json_data = self._load_data(input_settings.filepath)
            if input_settings.ignore_id:
                self._plot_tracks_scatter(ax, json_data)
            else:
                self._plot_tracks(ax, json_data)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(input_settings.filepath.stem)

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
