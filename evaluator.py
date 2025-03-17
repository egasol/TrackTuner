import json
import numpy as np
import statistics
import argparse
from pathlib import Path
from tabulate import tabulate
from typing import Any, Dict, Set

import utilities


class Statistics:
    def __init__(self) -> None:
        self.annotation_stats: Dict[int, Dict[str, Any]] = {}
        self.track_stats: Dict[int, Dict[str, Any]] = {}
        self.false_positives: int = 0

    def add_annotation(self, frame: int, obj_id: int, obj_position: np.ndarray) -> None:
        if obj_id not in self.annotation_stats:
            self.annotation_stats[obj_id] = {
                "lifespan": 0,
                "tracked": 0,
                "id_switches": 0,
                "last_track_id": None,
                "associated_track_ids": set(),
                "track_id_count": {},
            }
        self.annotation_stats[obj_id]["lifespan"] += 1

    def add_track(
        self,
        frame: int,
        obj_id: int,
        track_id: int,
        track_position: np.ndarray,
        obj_position: np.ndarray,
    ) -> bool:
        distance = np.linalg.norm(track_position - obj_position)

        if distance <= 4:
            if track_id not in self.track_stats:
                self.track_stats[track_id] = {
                    "lifespan": 0,
                    "tracked": 0,
                    "id_switches": 0,
                    "last_obj_id": None,
                    "associated_obj_ids": set(),
                }

            self.track_stats[track_id]["lifespan"] += 1
            self.track_stats[track_id]["associated_obj_ids"].add(obj_id)
            self.annotation_stats[obj_id]["associated_track_ids"].add(track_id)

            if self.annotation_stats[obj_id]["last_track_id"] != track_id:
                self.annotation_stats[obj_id]["id_switches"] += 1
                self.annotation_stats[obj_id]["last_track_id"] = track_id

            self.annotation_stats[obj_id]["tracked"] += 1
            self.track_stats[track_id]["tracked"] += 1

            if track_id not in self.annotation_stats[obj_id]["track_id_count"]:
                self.annotation_stats[obj_id]["track_id_count"][track_id] = 0
            self.annotation_stats[obj_id]["track_id_count"][track_id] += 1

            return True
        return False

    def update_false_positives(self, track_id: int) -> None:
        self.false_positives += 1

    def calculate_statistics(self) -> None:
        for obj_id, stats in self.annotation_stats.items():
            if stats["track_id_count"]:
                longest_match_track_id = max(
                    stats["track_id_count"], key=stats["track_id_count"].get
                )
                longest_match_count = stats["track_id_count"][longest_match_track_id]
                tracked_percentage = (longest_match_count / stats["lifespan"]) * 100
            else:
                tracked_percentage = 0
            stats["successfully_tracked"] = tracked_percentage >= 75
            stats["tracked_percentage"] = tracked_percentage

    def get_performance_metric(self) -> float:
        alpha = -5
        beta = 10
        gamma = 3.5

        tracked_percentage = statistics.mean(
            [stats["tracked_percentage"] for stats in self.annotation_stats.values()]
        )
        id_switches = statistics.mean(
            [stats["id_switches"] for stats in self.annotation_stats.values()]
        )

        performance_metric = (
            alpha * tracked_percentage
            + beta * id_switches
            + gamma * self.false_positives
        )
        return performance_metric

    def get_performance_multi_metric(self) -> (float, float, int):
        tracked_percentages = (
            stats["tracked_percentage"] for stats in self.annotation_stats.values()
        )
        id_switches = (stats["id_switches"] for stats in self.annotation_stats.values())

        average_tracked_percentage = statistics.mean(tracked_percentages)
        average_id_switches = statistics.mean(id_switches)

        return average_tracked_percentage, average_id_switches, self.false_positives

    def print_statistics(self) -> None:
        annotation_table = []
        for obj_id, stats in self.annotation_stats.items():
            annotation_table.append(
                [
                    obj_id,
                    stats["lifespan"],
                    stats["tracked"],
                    f"{stats['tracked_percentage']:.2f}%",
                    stats["id_switches"],
                    stats["successfully_tracked"],
                    ", ".join(map(str, sorted(stats["associated_track_ids"]))),
                ]
            )

        if annotation_table:
            annotation_headers = [
                "Object ID",
                "Lifespan (frames)",
                "Tracked (frames)",
                "Tracked %",
                "ID Switches",
                "Successfully Tracked",
                "Associated Track IDs",
            ]
            print("\nReference based statistics")
            print(
                tabulate(annotation_table, headers=annotation_headers, tablefmt="grid")
            )

        track_table = []
        for track_id, stats in self.track_stats.items():
            track_table.append(
                [
                    track_id,
                    stats["lifespan"],
                    ", ".join(map(str, sorted(stats["associated_obj_ids"]))),
                ]
            )

        if track_table:
            track_headers = ["Track ID", "Lifespan (frames)", "Associated Object IDs"]
            print("\nTrack based statistics")
            print(
                tabulate(
                    track_table,
                    headers=track_headers,
                    tablefmt="grid",
                    colalign=("center", "center", "left"),
                )
            )


def process_data(annotations: Dict[str, Any], tracks: Dict[str, Any]) -> Statistics:
    stats = Statistics()
    all_tracks = set()
    matched_tracks = set()

    for frame, annotation in annotations.items():
        for obj in annotation["tracks"]:
            obj_id = obj["id"]
            obj_position = np.array([obj["x"], obj["y"], obj["z"]])
            stats.add_annotation(frame, obj_id, obj_position)

            for track in tracks.get(frame, {}).get("tracks", []):
                track_id = track["id"]
                all_tracks.add(track_id)
                track_position = np.array([track["x"], track["y"], track["z"]])
                if stats.add_track(
                    frame, obj_id, track_id, track_position, obj_position
                ):
                    matched_tracks.add(track_id)

    for track_id in all_tracks:
        if track_id not in matched_tracks:
            stats.update_false_positives(track_id)

    stats.calculate_statistics()
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate references with tracked objects."
    )

    parser.add_argument(
        "--input-references",
        type=Path,
        help="Path to references file.",
    )
    parser.add_argument(
        "--input-tracked",
        type=Path,
        help="Path to tracked objects file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotations = utilities.load_json(args.input_references)
    tracked = utilities.load_json(args.input_tracked)

    stats = process_data(annotations, tracked)
    stats.print_statistics()

    performance_metric = stats.get_performance_metric()
    avg_tracked_percentage, id_switches, false_positives = (
        stats.get_performance_multi_metric()
    )

    metrics = [
        ["Average tracked percentage", f"{avg_tracked_percentage:.2f}%"],
        ["Average ID switches", id_switches],
        ["False positives", false_positives],
        ["Single value performance metric", f"{performance_metric:.2f}"],
    ]

    print("Summary")
    print(
        tabulate(
            metrics,
            headers=["Metric", "Value"],
            tablefmt="grid",
            colalign=("left", "right"),
        )
    )


if __name__ == "__main__":
    main()
