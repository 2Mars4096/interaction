import pytest

from interaction.contracts import BoundingBox, GazeObservation, GroundedTarget
from interaction.vision.stability import StableGazeLock


def observation(x=0.15, y=0.15, confidence=0.95):
    return GazeObservation(x_norm=x, y_norm=y, confidence=confidence, fixation_ms=10000)


def target(target_id="upper_left", confidence=0.95):
    return GroundedTarget(target_id=target_id, label=target_id, role="region", confidence=confidence)


def lock_target(lock):
    assert not lock.update(observation(), target(), observed_at_ms=0)
    assert not lock.update(observation(), target(), observed_at_ms=60)
    assert lock.update(observation(), target(), observed_at_ms=120)


def test_lock_requires_both_multiple_samples_and_measured_duration():
    lock = StableGazeLock()
    assert not lock.update(observation(), target(), observed_at_ms=0)
    # A provider's nominal fixation_ms value is not evidence of elapsed time.
    assert not lock.update(observation(), target(), observed_at_ms=1)
    assert not lock.update(observation(), target(), observed_at_ms=2)
    assert lock.sample_count == 3 and lock.duration_ms == 2
    assert lock.update(observation(), target(), observed_at_ms=120)
    assert lock.sample_count == 4 and lock.duration_ms == 120

    lock.reset()
    assert not lock.update(observation(), target(), observed_at_ms=200)
    assert not lock.update(observation(), target(), observed_at_ms=320)
    assert lock.update(observation(), target(), observed_at_ms=321)


def test_duplicate_timestamps_add_samples_but_no_duration():
    lock = StableGazeLock()
    for _ in range(5):
        assert not lock.update(observation(), target(), observed_at_ms=0)
    assert lock.duration_ms == 0
    assert lock.sample_count == 5
    explicit_zero_duration = StableGazeLock(min_samples=2, min_stable_ms=0)
    assert not explicit_zero_duration.update(observation(), target(), observed_at_ms=0)
    assert explicit_zero_duration.update(observation(), target(), observed_at_ms=0)


def test_reset_invalidates_revision_once_and_discards_all_evidence():
    lock = StableGazeLock()
    original_revision = lock.revision
    lock.reset()
    assert lock.revision == original_revision
    lock_target(lock)
    lock.reset()
    assert lock.revision > original_revision
    assert lock.sample_count == 0 and lock.duration_ms == 0
    invalidated_revision = lock.revision
    lock.reset()
    assert lock.revision == invalidated_revision
    assert not lock.update(observation(), target(), observed_at_ms=180)


@pytest.mark.parametrize(
    "invalid_observation,invalid_target",
    [
        (None, target()),
        (observation(), None),
        (observation(confidence=0.59), target()),
        (observation(), target(confidence=0.59)),
        (GazeObservation.model_construct(x_norm=float("nan"), y_norm=0.15, confidence=0.95), target()),
        (GazeObservation.model_construct(x_norm=0.15, y_norm=float("inf"), confidence=0.95), target()),
        (GazeObservation.model_construct(x_norm=1.01, y_norm=0.15, confidence=0.95), target()),
        (GazeObservation.model_construct(x_norm=-0.01, y_norm=0.15, confidence=0.95), target()),
        (GazeObservation.model_construct(x_norm=0.15, y_norm=0.15, confidence=float("nan")), target()),
        (GazeObservation(target_id="upper_left", confidence=0.95), target()),
        (GazeObservation(target_id="other", x_norm=0.15, y_norm=0.15, confidence=0.95), target()),
    ],
)
def test_invalid_or_weak_evidence_immediately_removes_old_lock(invalid_observation, invalid_target):
    lock = StableGazeLock()
    lock_target(lock)
    revision = lock.revision
    assert not lock.update(invalid_observation, invalid_target, observed_at_ms=180)
    assert lock.sample_count == 0 and lock.duration_ms == 0
    assert lock.revision > revision
    assert not lock.update(observation(), target(), observed_at_ms=240)


@pytest.mark.parametrize("timestamp", [None, float("nan"), float("inf"), -1, 119, True, "180"])
def test_invalid_or_backward_timestamp_cannot_credit_fixation(timestamp):
    lock = StableGazeLock()
    lock_target(lock)
    revision = lock.revision
    assert not lock.update(observation(), target(), observed_at_ms=timestamp)
    assert lock.revision > revision
    assert lock.sample_count == 0


def test_long_gap_starts_new_sequence_instead_of_reusing_two_old_samples():
    lock = StableGazeLock()
    assert not lock.update(observation(), target(), observed_at_ms=0)
    assert not lock.update(observation(), target(), observed_at_ms=60)
    revision = lock.revision
    assert not lock.update(observation(), target(), observed_at_ms=400)
    assert lock.sample_count == 1 and lock.duration_ms == 0
    assert lock.revision > revision
    assert not lock.update(observation(), target(), observed_at_ms=460)
    assert lock.update(observation(), target(), observed_at_ms=520)


def test_raw_point_jitter_within_one_coarse_region_never_locks():
    lock = StableGazeLock()
    for index, x in enumerate([0.10, 0.25] * 5):
        assert not lock.update(observation(x=x), target(), observed_at_ms=index * 60)
        assert lock.sample_count == 1


def test_slow_drift_is_measured_from_sequence_anchor_not_previous_frame():
    lock = StableGazeLock()
    assert not lock.update(observation(x=0.10), target(), observed_at_ms=0)
    assert not lock.update(observation(x=0.14), target(), observed_at_ms=60)
    assert not lock.update(observation(x=0.18), target(), observed_at_ms=120)
    assert lock.sample_count == 1


def test_small_raw_variation_can_lock_without_altering_confidence():
    lock = StableGazeLock()
    assert not lock.update(observation(x=0.15), target(), observed_at_ms=0)
    assert not lock.update(observation(x=0.17), target(), observed_at_ms=60)
    assert lock.update(observation(x=0.16), target(), observed_at_ms=120)


def test_away_and_back_cannot_restore_original_revision():
    lock = StableGazeLock()
    lock_target(lock)
    original = lock.revision
    assert not lock.update(observation(x=0.5), target("upper_middle"), observed_at_ms=180)
    assert not lock.update(observation(), target(), observed_at_ms=240)
    assert not lock.update(observation(), target(), observed_at_ms=300)
    assert lock.update(observation(), target(), observed_at_ms=360)
    assert lock.revision > original


def test_target_movement_with_same_id_invalidates_old_revision():
    lock = StableGazeLock()
    lock_target(lock)
    original = lock.revision
    moved = target().model_copy(update={"bounds": BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)})
    assert not lock.update(observation(), moved, observed_at_ms=180)
    assert lock.revision > original and lock.sample_count == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"min_samples": 1}, {"min_samples": True}, {"min_stable_ms": -1},
        {"min_stable_ms": float("nan")}, {"max_gap_ms": 0}, {"max_gap_ms": float("inf")},
        {"max_radius": -0.01}, {"min_confidence": 1.1},
    ],
)
def test_invalid_lock_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        StableGazeLock(**settings)
