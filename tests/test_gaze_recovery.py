import pytest

from interaction.contracts import ActionName, EnvironmentSnapshot, GazeObservation
from interaction.runtime import GazeTrackingLoop
from interaction.vision import (
    CalibrationProfile,
    DwellTrigger,
    GazeSample,
    GazeSmoother,
    GazeTargetInferencer,
    NormalizedPoint,
    NormalizedScreenTarget,
)


TARGETS = [
    NormalizedScreenTarget("left", "Left", "region", 0.0, 0.0, 0.45, 1.0),
    NormalizedScreenTarget("right", "Right", "region", 0.55, 0.0, 0.45, 1.0),
]
ENVIRONMENT = EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")


def observation(x=0.2, *, elapsed_ms=100, confidence=0.9):
    return GazeObservation(x_norm=x, y_norm=0.5, confidence=confidence, fixation_ms=elapsed_ms)


def make_loop(*, action=ActionName.CLICK_TARGET, window_size=4, drag_timeout_ms=3000):
    return GazeTrackingLoop(
        smoother=GazeSmoother(window_size=window_size),
        dwell_trigger=DwellTrigger(dwell_ms=300, action=action, cooldown_ms=200, rearm_ms=200),
        auto_confirm_actions={action}, drag_timeout_ms=drag_timeout_ms,
    )


def sample(
    loop, x=0.2, *, confidence=0.9, elapsed_ms=100,
    environment=ENVIRONMENT, targets=TARGETS, measured_timing=False,
):
    return loop.process_sample(
        GazeSample(NormalizedPoint(x, 0.5), confidence=confidence, delta_ms=elapsed_ms), targets, environment,
        measured_timing=measured_timing,
    )


def results(events):
    return [event.result for event in events if event.result is not None]


def test_smoother_reset_removes_previous_target_points():
    smoother = GazeSmoother(window_size=4)
    smoother.smooth(GazeSample(NormalizedPoint(0.1, 0.1)))
    smoother.reset()

    current = smoother.smooth(GazeSample(NormalizedPoint(0.9, 0.8)))

    assert current.x_norm == 0.9
    assert current.y_norm == 0.8


def test_target_confidence_is_detection_evidence_without_inflation():
    target = GazeTargetInferencer().infer(observation(confidence=0.6), TARGETS)

    assert target is not None
    assert target.confidence == 0.6
    assert GazeTargetInferencer().infer(observation(confidence=0.59), TARGETS) is None


@pytest.mark.parametrize("field, value", [
    ("x_norm", float("nan")), ("y_norm", float("inf")),
    ("x_norm", -0.1), ("y_norm", 1.1), ("x_norm", None),
    ("confidence", float("nan")), ("confidence", float("inf")), ("confidence", True),
])
def test_inferencer_rejects_invalid_observations_even_without_contract_validation(field, value):
    values = {"x_norm": 0.2, "y_norm": 0.5, "confidence": 0.9, "fixation_ms": 100}
    values[field] = value

    assert GazeTargetInferencer().infer(GazeObservation.model_construct(**values), TARGETS) is None


def test_same_target_never_repeats_after_action_cooldown_expires():
    loop = make_loop()
    events = [event for _ in range(50) for event in sample(loop)]

    assert len(results(events)) == 1
    assert loop.dwell_trigger.triggered_target_id == "left"
    assert loop.dwell_trigger.cooldown_remaining_ms == 0


def test_single_observed_lookaway_cannot_rearm_using_incoming_frame_interval():
    loop = make_loop()
    loop.dwell_trigger.dwell_ms = 700
    loop.dwell_trigger.cooldown_ms = 0

    events = [
        event for x in [0.2] * 5 + [0.8] + [0.2] * 4
        for event in sample(loop, x, elapsed_ms=200, measured_timing=True)
    ]

    assert len(results(events)) == 1
    assert loop.dwell_trigger.triggered_target_id == "left"


def test_measured_rearming_requires_observed_away_time_then_fresh_target_dwell():
    loop = make_loop()
    loop.dwell_trigger.dwell_ms = 700
    loop.dwell_trigger.cooldown_ms = 0
    for _ in range(5):
        sample(loop, elapsed_ms=200, measured_timing=True)
    for _ in range(3):
        assert results(sample(loop, 0.8, elapsed_ms=200, measured_timing=True)) == []
    assert loop.dwell_trigger.triggered_target_id is None

    # Returning to the original target establishes a new starting point; the
    # incoming interval belongs to neither its new dwell nor the look-away.
    assert results(sample(loop, elapsed_ms=200, measured_timing=True)) == []
    assert loop.dwell_trigger.accumulated_ms == 0
    for _ in range(3):
        assert results(sample(loop, elapsed_ms=200, measured_timing=True)) == []
    assert len(results(sample(loop, elapsed_ms=200, measured_timing=True))) == 1


def test_measured_target_transition_ages_cooldown_without_crediting_fixation():
    loop = make_loop()
    loop.dwell_trigger.cooldown_ms = 600
    for _ in range(4):
        sample(loop, measured_timing=True)
    assert loop.dwell_trigger.cooldown_remaining_ms == 600

    events = sample(loop, 0.8, elapsed_ms=200, measured_timing=True)

    assert loop.dwell_trigger.cooldown_remaining_ms == 400
    assert loop.dwell_trigger.lookaway_ms == 0
    assert events[-1].observation.fixation_ms == 0


def test_loss_and_same_target_reacquisition_cannot_repeat_fired_action():
    loop = make_loop()
    assert len(results([event for _ in range(3) for event in sample(loop)])) == 1
    loop.tracking_lost(elapsed_ms=5000)

    events = [event for _ in range(30) for event in sample(loop)]

    assert results(events) == []
    assert loop.dwell_trigger.triggered_target_id == "left"


def test_new_target_requires_lookaway_then_a_complete_fresh_dwell():
    trigger = DwellTrigger(dwell_ms=300, cooldown_ms=0, rearm_ms=200)
    left, right = [target.to_grounded_target(0.9) for target in TARGETS]
    for _ in range(2):
        assert trigger.update(observation(), left) is None
    assert trigger.update(observation(), left) is not None

    assert trigger.update(observation(0.8), right) is None
    assert trigger.update(observation(0.8), right) is None
    assert trigger.triggered_target_id is None
    assert trigger.accumulated_ms == 0
    assert trigger.update(observation(0.8), right) is None
    assert trigger.update(observation(0.8), right) is None
    assert trigger.update(observation(0.8), right) is not None


def test_interrupted_lookaway_does_not_rearm_fired_target():
    trigger = DwellTrigger(dwell_ms=300, cooldown_ms=0, rearm_ms=200)
    left = TARGETS[0].to_grounded_target(0.9)
    for _ in range(3):
        trigger.update(observation(), left)
    trigger.update(observation(0.5), None)
    assert trigger.lookaway_ms == 100

    trigger.reset(elapsed_ms=5000)
    trigger.update(observation(0.5), None)
    assert trigger.lookaway_ms == 100
    assert trigger.triggered_target_id == "left"
    assert all(trigger.update(observation(), left) is None for _ in range(20))


def test_valid_background_lookaway_rearms_without_being_treated_as_tracking_loss():
    loop = make_loop()
    assert len(results([event for _ in range(3) for event in sample(loop)])) == 1
    assert results(sample(loop, 0.5)) == []
    assert results(sample(loop, 0.5)) == []
    assert loop.dwell_trigger.triggered_target_id is None

    events = [event for _ in range(3) for event in sample(loop)]

    assert len(results(events)) == 1


@pytest.mark.parametrize("interruption", ["missing", "weak", "no_target"])
def test_interruption_clears_partial_dwell_and_smoothing(interruption):
    loop = make_loop()
    sample(loop)
    sample(loop)
    assert loop.dwell_trigger.accumulated_ms == 200
    if interruption == "missing":
        loop.tracking_lost(elapsed_ms=5000)
    elif interruption == "weak":
        sample(loop, confidence=0.1)
    else:
        sample(loop, 0.5)
    assert loop.dwell_trigger.accumulated_ms == 0
    assert not loop.smoother.window

    assert results(sample(loop)) == []
    assert results(sample(loop)) == []
    additional_samples = 1 if interruption == "no_target" else 2
    events = [event for _ in range(additional_samples) for event in sample(loop)]
    assert len(results(events)) == 1


def test_first_reacquired_reading_never_credits_missing_time_as_dwell():
    loop = make_loop()
    sample(loop)
    loop.tracking_lost(elapsed_ms=10_000)

    events = sample(loop, 0.8, elapsed_ms=10_000)

    assert results(events) == []
    assert loop.dwell_trigger.accumulated_ms == 0
    assert events[-1].observation.x_norm == 0.8
    assert events[-1].observation.fixation_ms == 0


def test_wandering_within_a_coarse_target_cannot_hide_behind_smoothing():
    loop = make_loop(window_size=20)
    large = [NormalizedScreenTarget("large", "Large", "region", 0.0, 0.0, 1.0, 1.0)]
    events = [event for x in [0.2, 0.4] * 20 for event in sample(loop, x, targets=large)]

    assert results(events) == []
    assert loop.dwell_trigger.accumulated_ms == 0


def test_moving_within_fired_target_cannot_rearm_it():
    loop = make_loop()
    large = [NormalizedScreenTarget("large", "Large", "region", 0.0, 0.0, 1.0, 1.0)]
    assert len(results([event for _ in range(3) for event in sample(loop, targets=large)])) == 1

    events = [event for x in [0.8, 0.2] * 20 for event in sample(loop, x, targets=large)]

    assert results(events) == []
    assert loop.dwell_trigger.triggered_target_id == "large"


@pytest.mark.parametrize("raw_x", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_raw_coordinates_cannot_be_clamped_into_edge_actions(raw_x):
    loop = make_loop()

    events = sample(loop, raw_x, elapsed_ms=1000)

    assert results(events) == []
    assert loop.dwell_trigger.accumulated_ms == 0
    assert all(event.target is None for event in events)


def test_out_of_screen_calibration_projection_cannot_be_clamped_into_edge_actions():
    loop = make_loop()
    loop.calibration_profile = CalibrationProfile(x_scale=2)

    events = [event for _ in range(5) for event in sample(loop, 0.8, elapsed_ms=1000)]

    assert results(events) == []
    assert all(event.target is None for event in events)
    assert not loop.smoother.window


@pytest.mark.parametrize("interruption", ["missing", "weak", "no_target"])
def test_tracking_interruption_cancels_armed_drag_before_reacquisition(interruption):
    loop = make_loop(action=ActionName.DRAG_TARGET)
    for _ in range(3):
        sample(loop)
    assert loop.pending_drag_origin is not None
    if interruption == "missing":
        events = loop.tracking_lost(elapsed_ms=5000)
    elif interruption == "weak":
        events = sample(loop, confidence=0.1)
    else:
        events = sample(loop, 0.5)

    assert loop.pending_drag_origin is None
    assert any("cancelled" in event.message for event in events)
    assert all(event.target is None for event in events)
    # Reacquisition can eventually arm a new source, but cannot use the lost
    # source to execute a drag merely by dwelling at the former destination.
    events.extend(event for _ in range(10) for event in sample(loop, 0.8))
    assert results(events) == []
    assert loop.pending_drag_origin.target.target_id == "right"


@pytest.mark.parametrize("changed_environment", [
    EnvironmentSnapshot(active_app="Terminal", active_window_title="Inbox"),
    EnvironmentSnapshot(active_app="Mail", active_window_title="Compose"),
    EnvironmentSnapshot(),
])
def test_foreground_context_change_cancels_armed_drag(changed_environment):
    loop = make_loop(action=ActionName.DRAG_TARGET)
    for _ in range(3):
        sample(loop)
    assert loop.pending_drag_origin is not None

    events = sample(loop, 0.8, environment=changed_environment)

    assert loop.pending_drag_origin is None
    assert any("application or window changed" in event.message for event in events)
    events.extend(event for _ in range(10) for event in sample(loop, 0.8, environment=changed_environment))
    assert results(events) == []


def test_drag_environment_is_copied_when_origin_is_armed():
    loop = make_loop(action=ActionName.DRAG_TARGET)
    environment = EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")
    for _ in range(3):
        sample(loop, environment=environment)
    environment.active_app = "Terminal"

    events = sample(loop, 0.8, environment=environment)

    assert loop.pending_drag_origin is None
    assert any("application or window changed" in event.message for event in events)


def test_drag_timeout_still_expires_with_continuous_valid_gaze_at_source():
    loop = make_loop(action=ActionName.DRAG_TARGET, drag_timeout_ms=600)
    for _ in range(3):
        sample(loop)
    assert loop.pending_drag_origin is not None

    events = [event for _ in range(6) for event in sample(loop)]

    assert loop.pending_drag_origin is None
    assert any("timed out" in event.message for event in events)
    assert results(events) == []
