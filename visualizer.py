import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from time import time
from pathlib import Path
from typing import Dict, Any

from utilities import load_json


class VisualizerInput:
    def __init__(self, filepath: Path, title: str, ignore_id: bool = False):
        self.filepath = filepath
        self.title = title
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
        return load_json(filepath)

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

    def visualize(self, output: Path, dpi: int = 50) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)

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
            ax.set_title(input_settings.title)

        plt.savefig(output, dpi=dpi)
        print("Summarizing plots to", output, f"({time() - t0:.2f}s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tracker given detections and tracker parameters."
    )

    parser.add_argument(
        "--input-references",
        type=Path,
        required=False,
        default=None,
        help="Path to detections json file.",
    )
    parser.add_argument(
        "--input-detections",
        type=Path,
        required=False,
        default=None,
        help="Path to detections json file.",
    )
    parser.add_argument(
        "--input-tracked",
        type=Path,
        required=False,
        default=None,
        help="Path to tracked json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output visualization comparison.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_files = [
        VisualizerInput(path, title=title, ignore_id=ignore_id)
        for path, title, ignore_id in [
            (args.input_references, "references", False),
            (args.input_detections, "detections", True),
            (args.input_tracked, "tracked", False),
        ]
        if path is not None
    ]

    assert len(input_files) > 0, "Error: Please specify at least one input file."

    visualizer = Visualizer(input_files)
    visualizer.visualize(args.output)


if __name__ == "__main__":
    main()
