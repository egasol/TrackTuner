import pytest
import numpy as np
from tracker import TrackSettings, Track, Tracker, TrackStage


@pytest.fixture
def track_settings():
    return TrackSettings(
        measurement_noise=0.1,
        process_noise=0.1,
        covariance=1.0,
        distance_threshold=2.0,
        max_age=10,
        min_hits=3,
        max_consecutive_misses=5,
    )


@pytest.fixture
def initial_position():
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture
def initial_velocity():
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def initial_acceleration():
    return np.array([0.1, 0.1, 0.1])


@pytest.fixture
def track(track_settings, initial_position, initial_velocity, initial_acceleration):
    return Track(
        1, initial_position, initial_velocity, initial_acceleration, track_settings
    )


def test_track_initialization(
    track, initial_position, initial_velocity, initial_acceleration
):
    assert track.id == 1
    assert np.allclose(track.get_state(), initial_position)
    assert np.allclose(track.get_velocity(), initial_velocity)
    assert np.allclose(track.get_acceleration(), initial_acceleration)
    assert track.stage == TrackStage.INITIALIZED
    assert track.age == 0
    assert track.hits == 1
    assert track.hit_streak == 0
    assert track.time_since_update == 0
    assert track.consecutive_misses == 0
    assert np.array_equal(track.position_history, [initial_position.tolist()])


def test_track_prediction(track):
    predicted_state = track.predict()
    assert track.age == 1
    assert track.consecutive_misses == 1
    assert np.allclose(predicted_state, track.kf.x)


def test_track_update(track):
    new_measurement = np.array([1.0, 1.0, 1.0])
    track.update(new_measurement)
    assert track.time_since_update == 0
    assert track.hits == 2
    assert track.hit_streak == 1
    assert track.consecutive_misses == 0
    assert np.allclose(track.position_history[-1], new_measurement)


def test_tracker_initialization(track_settings):
    tracker = Tracker(track_settings)
    assert tracker.tracks == []
    assert tracker.track_id == 0
    assert tracker.distance_threshold == track_settings.distance_threshold
    assert tracker.max_age == track_settings.max_age
    assert tracker.min_hits == track_settings.min_hits
    assert tracker.max_consecutive_misses == track_settings.max_consecutive_misses


def test_tracker_associate_detections_to_tracks(
    track_settings, initial_position, initial_velocity, initial_acceleration
):
    tracker = Tracker(track_settings)
    track = Track(
        1, initial_position, initial_velocity, initial_acceleration, track_settings
    )
    tracker.tracks.append(track)
    detections = [np.array([1.0, 1.0, 1.0])]
    assigned_tracks, unassigned_tracks, unassigned_detections = (
        tracker.associate_detections_to_tracks(detections)
    )
    assert assigned_tracks == [0]
    assert unassigned_tracks == []
    assert unassigned_detections == []


# def test_tracker_update_tracks(track_settings, initial_position, initial_velocity, initial_acceleration):
# 	tracker = Tracker(track_settings)
# 	detections = [initial_position]
# 	tracker.update_tracks(detections)
# 	assert len(tracker.tracks) == 1
# 	assert tracker.tracks[0].get_state().tolist() == initial_position.tolist()
# 	detections = [np.array([1.0, 1.0, 1.0])]
# 	tracker.update_tracks(detections)
# 	assert len(tracker.tracks) == 1
# 	state_after_second_update = tracker.tracks[0].get_state()

# 	# Ensure the state has moved closer to the new detection
# 	assert np.linalg.norm(state_after_second_update - new_detection) <
# 	# assert tracker.tracks[0].get_state().tolist() == detections[0].tolist()
# 	# assert tracker.


def test_tracker_update_tracks(
    track_settings, initial_position, initial_velocity, initial_acceleration
):
    tracker = Tracker(track_settings)
    detections = [initial_position]
    tracker.update_tracks(detections)

    assert len(tracker.tracks) == 1

    # Initial update, state should be close to the initial position
    state_after_first_update = tracker.tracks[0].get_state()
    assert np.allclose(state_after_first_update, initial_position, atol=1e-1)

    # Introduce a new detection and update tracks
    new_detection = np.array([1.0, 1.0, 1.0])
    tracker.update_tracks([new_detection])

    state_after_second_update = tracker.tracks[0].get_state()

    # Ensure the state has moved closer to the new detection
    assert np.linalg.norm(state_after_second_update - new_detection) < np.linalg.norm(
        state_after_first_update - new_detection
    )
