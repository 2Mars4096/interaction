import pytest

from interaction.contracts import EnvironmentSnapshot
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
