import pytest
import numpy as np

from annotator import TrackGenerator
from datatypes.reference import Reference, ReferenceTrack
from datatypes.detection import Detection


@pytest.fixture
def track_generator():
    return TrackGenerator(
        num_frames=100,
        num_tracks=5,
        position_randomization=0.1,
        delete_probability=1.0,
        add_probability=0.0,
    )


def test_generate_annotations(track_generator):
    annotations = track_generator.generate_annotations()
    assert isinstance(annotations, dict)
    assert len(annotations) == track_generator.num_frames
    for frame, tracks in annotations.items():
        assert isinstance(frame, int)
        assert isinstance(tracks, list)
        for track in tracks:
            assert isinstance(track, Reference)


def test_get_min_max_ranges(track_generator):
    min_max_ranges = track_generator.get_min_max_ranges()
    assert isinstance(min_max_ranges, dict)
    assert "x" in min_max_ranges
    assert "y" in min_max_ranges
    assert "z" in min_max_ranges
    for key, value in min_max_ranges.items():
        assert isinstance(value, tuple)
        assert len(value) == 2
        assert isinstance(value[0], float)
        assert isinstance(value[1], float)


def test_generate_false_positives(track_generator):
    new_tracks = []
    false_positives = track_generator.generate_false_positives(new_tracks)
    assert isinstance(false_positives, list)
    for track in false_positives:
        assert isinstance(track, Detection)


def test_modify_tracks(track_generator):
    modified_tracks = track_generator.modify_tracks()
    assert isinstance(modified_tracks, dict)
    assert len(modified_tracks) == track_generator.num_frames
    for frame, tracks in modified_tracks.items():
        assert isinstance(frame, int)
        assert isinstance(tracks, list)
        for track in tracks:
            assert isinstance(track, Detection)


if __name__ == "__main__":
    pytest.main()
