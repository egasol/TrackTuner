import json
import numpy as np
import statistics
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
            return True
        return False

    def update_false_positives(self, track_id: int) -> None:
        self.false_positives += 1

    def calculate_statistics(self) -> None:
        for obj_id, stats in self.annotation_stats.items():
            tracked_percentage = (stats["tracked"] / stats["lifespan"]) * 100
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
        print("Tracking Performance Statistics:")
        print("Annotations:")
        for obj_id, stats in self.annotation_stats.items():
            print(f"Object ID: {obj_id}")
            print(f"  Lifespan: {stats['lifespan']} frames")
            print(
                f"  Tracked: {stats['tracked']} frames ({stats['tracked_percentage']:.2f}%)"
            )
            print(f"  ID Switches: {stats['id_switches']}")
            print(f"  Successfully Tracked: {stats['successfully_tracked']}")
            print(f"  Associated Track IDs: {sorted(stats['associated_track_ids'])}")

        print("\nTracks:")
        for track_id, stats in self.track_stats.items():
            print(f"Track ID: {track_id}")
            print(f"  Lifespan: {stats['lifespan']} frames")
            print(f"  Tracked: {stats['tracked']} frames")
            print(f"  ID Switches: {stats['id_switches']}")
            print(f"  Associated Object IDs: {sorted(stats['associated_obj_ids'])}")


def process_data(annotations: Dict[str, Any], tracks: Dict[str, Any]) -> Statistics:
    stats = Statistics()

    for frame, annotation in annotations.items():
        tracked_objs: Set[int] = set()
        for obj in annotation["tracks"]:
            obj_id = obj["id"]
            obj_position = np.array([obj["x"], obj["y"], obj["z"]])
            stats.add_annotation(frame, obj_id, obj_position)

            for track in tracks.get(frame, {}).get("tracks", []):
                track_id = track["id"]
                track_position = np.array([track["x"], track["y"], track["z"]])
                if stats.add_track(
                    frame, obj_id, track_id, track_position, obj_position
                ):
                    tracked_objs.add(track_id)

        for track in tracks.get(frame, {}).get("tracks", []):
            if track["id"] not in tracked_objs:
                stats.update_false_positives(track["id"])

    stats.calculate_statistics()
    return stats


def main() -> None:
    tracked = utilities.load_json(utilities.get_data_path() / "tracked.json")
    annotations = utilities.load_json(utilities.get_data_path() / "annotations.json")

    stats = process_data(annotations, tracked)
    stats.print_statistics()
    print("performance_metric:", stats.get_performance_metric())
    print("performance_multi_metrics:", stats.get_performance_multi_metric())


if __name__ == "__main__":
    main()
