import pytest
import numpy as np
from evaluator import Statistics, process_data


@pytest.fixture
def stats():
    return Statistics()


def test_add_annotation(stats):
    frame = 1
    obj_id = 42
    obj_position = np.array([0.0, 0.0, 0.0])
    stats.add_annotation(frame, obj_id, obj_position)
    assert obj_id in stats.annotation_stats
    assert stats.annotation_stats[obj_id]["lifespan"] == 1


def test_add_track(stats):
    frame = 1
    obj_id = 42
    track_id = 101
    obj_position = np.array([0.0, 0.0, 0.0])
    track_position = np.array([0.0, 0.0, 1.0])

    stats.add_annotation(frame, obj_id, obj_position)
    is_tracked = stats.add_track(frame, obj_id, track_id, track_position, obj_position)
    assert is_tracked is True
    assert track_id in stats.track_stats
    assert stats.track_stats[track_id]["lifespan"] == 1
    assert stats.track_stats[track_id]["tracked"] == 1


def test_update_false_positives(stats):
    track_id = 101
    initial_false_positives = stats.false_positives
    stats.update_false_positives(track_id)
    assert stats.false_positives == initial_false_positives + 1


def test_calculate_statistics(stats):
    frame = 1
    obj_id = 42
    obj_position = np.array([0.0, 0.0, 0.0])

    for _ in range(4):
        stats.add_annotation(frame, obj_id, obj_position)
        stats.add_track(frame, obj_id, obj_id, obj_position, obj_position)

    stats.calculate_statistics()
    assert stats.annotation_stats[obj_id]["tracked_percentage"] == 100.0
    assert stats.annotation_stats[obj_id]["successfully_tracked"] is True


def test_get_performance_metric(stats):
    frame = 1
    obj_id = 42
    obj_position = np.array([0.0, 0.0, 0.0])

    for _ in range(4):
        stats.add_annotation(frame, obj_id, obj_position)
        stats.add_track(frame, obj_id, obj_id, obj_position, obj_position)

    stats.calculate_statistics()
    performance_metric = stats.get_performance_metric()
    assert isinstance(performance_metric, float)


def test_process_data():
    annotations = {"1": {"tracks": [{"id": 42, "x": 0.0, "y": 0.0, "z": 0.0}]}}
    tracks = {"1": {"tracks": [{"id": 101, "x": 0.0, "y": 0.0, "z": 0.0}]}}
    stats = process_data(annotations, tracks)
    assert isinstance(stats, Statistics)
    assert stats.false_positives == 0
    assert stats.annotation_stats[42]["tracked_percentage"] == 100.0


if __name__ == "__main__":
    pytest.main()
