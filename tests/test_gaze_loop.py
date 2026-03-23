import pytest

from interaction.contracts import ActionName, EnvironmentSnapshot
from interaction.feedback import GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import GazeTrackingLoop
from interaction.vision import (
    CalibrationProfile,
    CalibrationSample,
    DwellTrigger,
    GazeSample,
    GazeSmoother,
    GazeTargetInferencer,
    NormalizedPoint,
    NormalizedScreenTarget,
)


def test_calibration_profile_adjusts_offsets() -> None:
    profile = CalibrationProfile.fit(
        [
            CalibrationSample(raw=NormalizedPoint(0.4, 0.4), expected=NormalizedPoint(0.5, 0.5)),
            CalibrationSample(raw=NormalizedPoint(0.6, 0.6), expected=NormalizedPoint(0.7, 0.7)),
        ]
    )

    adjusted = profile.apply(NormalizedPoint(0.4, 0.4))

    assert adjusted.x == 0.5
    assert adjusted.y == 0.5


def test_gaze_smoother_averages_points() -> None:
    smoother = GazeSmoother(window_size=2)
    smoother.smooth(GazeSample(point=NormalizedPoint(0.2, 0.2)))
    observation = smoother.smooth(GazeSample(point=NormalizedPoint(0.4, 0.4)))

    assert observation.x_norm == pytest.approx(0.3)
    assert observation.y_norm == pytest.approx(0.3)


def test_inferencer_prefers_large_target() -> None:
    inferencer = GazeTargetInferencer(min_width=0.1, min_height=0.1)
    targets = [
        NormalizedScreenTarget(target_id="tiny", label="Tiny", role="button", x=0.4, y=0.4, width=0.02, height=0.02),
        NormalizedScreenTarget(target_id="large", label="Large", role="button", x=0.3, y=0.3, width=0.3, height=0.3),
    ]
    observation = GazeSmoother(window_size=1).smooth(GazeSample(point=NormalizedPoint(0.45, 0.45), confidence=0.9))

    target = inferencer.infer(observation, targets)

    assert target is not None
    assert target.target_id == "large"


def test_dwell_trigger_fires_once_after_threshold() -> None:
    target = NormalizedScreenTarget(target_id="compose", label="Compose", role="button", x=0.5, y=0.2, width=0.2, height=0.1).to_grounded_target(0.9)
    trigger = DwellTrigger(dwell_ms=500, min_confidence=0.6)
    smoother = GazeSmoother(window_size=1)

    observation1 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=200))
    observation2 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=200))
    observation3 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=200))

    assert trigger.update(observation1, target) is None
    assert trigger.update(observation2, target) is None
    proposal = trigger.update(observation3, target)
    assert proposal is not None
    assert proposal.action.value == "highlight_target"
    assert trigger.update(observation3, target) is None


def test_dwell_trigger_click_mode_includes_normalized_point() -> None:
    target = NormalizedScreenTarget(target_id="compose", label="Compose", role="button", x=0.5, y=0.2, width=0.2, height=0.1).to_grounded_target(0.9)
    trigger = DwellTrigger(dwell_ms=500, min_confidence=0.6, action=ActionName.CLICK_TARGET)
    smoother = GazeSmoother(window_size=1)

    observation1 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=250))
    observation2 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=250))

    assert trigger.update(observation1, target) is None
    proposal = trigger.update(observation2, target)

    assert proposal is not None
    assert proposal.action == ActionName.CLICK_TARGET
    assert proposal.arguments["normalized_point"] == {"x": 0.55, "y": 0.22}
    assert proposal.arguments["target_bounds"]["width"] == pytest.approx(0.2)


def test_dwell_trigger_respects_action_cooldown_before_rearming() -> None:
    compose = NormalizedScreenTarget(target_id="compose", label="Compose", role="button", x=0.5, y=0.2, width=0.2, height=0.1).to_grounded_target(0.9)
    sidebar = NormalizedScreenTarget(target_id="sidebar", label="Sidebar", role="panel", x=0.02, y=0.1, width=0.2, height=0.7).to_grounded_target(0.9)
    trigger = DwellTrigger(dwell_ms=500, min_confidence=0.6, cooldown_ms=600)
    smoother = GazeSmoother(window_size=1)

    observation1 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=250))
    observation2 = smoother.smooth(GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.9, delta_ms=250))
    observation3 = smoother.smooth(GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.9, delta_ms=250))
    observation4 = smoother.smooth(GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.9, delta_ms=250))
    observation5 = smoother.smooth(GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.9, delta_ms=250))
    observation6 = smoother.smooth(GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.9, delta_ms=250))

    assert trigger.update(observation1, compose) is None
    assert trigger.update(observation2, compose) is not None
    assert trigger.update(observation3, sidebar) is None
    assert trigger.update(observation4, sidebar) is None
    assert trigger.update(observation5, sidebar) is None
    assert trigger.update(observation6, sidebar) is not None


def test_gaze_loop_triggers_highlight_dry_run() -> None:
    loop = GazeTrackingLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.5, y=0.2, width=0.2, height=0.12)
    ]
    samples = [
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.85, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.86, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.87, delta_ms=250),
    ]

    events = loop.run_trace(samples, targets, EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == GazeLoopPhase.TRIGGERED for event in events)
    assert any(event.result and event.result.status.value == "success" for event in events)


def test_gaze_loop_click_mode_auto_confirms_explicit_gaze_session() -> None:
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=True),
        dwell_trigger=DwellTrigger(dwell_ms=500, action=ActionName.CLICK_TARGET),
        auto_confirm_actions={ActionName.CLICK_TARGET},
    )
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.5, y=0.2, width=0.2, height=0.12)
    ]
    samples = [
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.85, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.86, delta_ms=250),
    ]

    events = loop.run_trace(samples, targets, EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == GazeLoopPhase.TRIGGERED for event in events)
    assert any(event.result and event.result.details["commands"][0][3] == "click-normalized" for event in events)


def test_gaze_loop_drag_mode_arms_then_executes_drag() -> None:
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=True),
        dwell_trigger=DwellTrigger(dwell_ms=500, action=ActionName.DRAG_TARGET),
        auto_confirm_actions={ActionName.DRAG_TARGET},
        smoother=GazeSmoother(window_size=1),
    )
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.5, y=0.2, width=0.2, height=0.12),
        NormalizedScreenTarget(target_id="sidebar", label="Mail sidebar", role="panel", x=0.02, y=0.1, width=0.20, height=0.75),
    ]
    samples = [
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.85, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.86, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.87, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.08, 0.45), confidence=0.88, delta_ms=250),
    ]

    events = loop.run_trace(samples, targets, EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == GazeLoopPhase.TRIGGERED and event.result is None for event in events)
    assert any(
        event.result and event.result.details["commands"][0][3] == "drag-normalized"
        for event in events
    )


def test_gaze_loop_drag_origin_times_out_when_destination_never_arrives() -> None:
    loop = GazeTrackingLoop(
        adapter=MacOSPlatformAdapter(dry_run=True),
        dwell_trigger=DwellTrigger(dwell_ms=500, action=ActionName.DRAG_TARGET),
        auto_confirm_actions={ActionName.DRAG_TARGET},
        smoother=GazeSmoother(window_size=1),
        drag_timeout_ms=600,
    )
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.5, y=0.2, width=0.2, height=0.12),
    ]
    samples = [
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.85, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.55, 0.22), confidence=0.86, delta_ms=250),
        GazeSample(point=NormalizedPoint(0.9, 0.9), confidence=0.87, delta_ms=350),
        GazeSample(point=NormalizedPoint(0.9, 0.9), confidence=0.88, delta_ms=350),
    ]

    events = loop.run_trace(samples, targets, EnvironmentSnapshot(active_app="Mail"))

    assert any(event.message.startswith("Drag origin armed") for event in events)
    assert any(event.message.startswith('Drag origin at "Compose button" timed out.') for event in events)
    assert all(not (event.result and event.result.details["commands"][0][3] == "drag-normalized") for event in events)
