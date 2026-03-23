from interaction.contracts import EnvironmentSnapshot, GazeObservation
from interaction.feedback import FusionLoopPhase
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime import MultimodalFusionLoop, SharedInteractionState
from interaction.vision import NormalizedScreenTarget


def _compose_target(confidence: float = 0.9):
    return NormalizedScreenTarget(
        target_id="compose",
        label="Compose button",
        role="button",
        x=0.52,
        y=0.16,
        width=0.22,
        height=0.12,
    ).to_grounded_target(confidence=confidence)


def test_shared_state_expires_stale_target() -> None:
    state = SharedInteractionState(gaze_context_window_ms=500)
    target = _compose_target()
    state.record_gaze(GazeObservation(confidence=0.9, x_norm=0.56, y_norm=0.20, fixation_ms=200), target)
    state.advance(600)

    assert state.latest_target() is None
    assert not state.latest_target_is_fresh()


def test_fusion_loop_open_this_uses_recent_gaze_target() -> None:
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    target = _compose_target()
    loop.update_gaze_context(GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=250), target)

    events = loop.run_voice_turn("open this", EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == FusionLoopPhase.CONFIRMING for event in events)
    assert events[-1].phase != FusionLoopPhase.IDLE or any(event.phase == FusionLoopPhase.CONFIRMING for event in events)


def test_fusion_loop_confirm_yes_executes_normalized_pointer_action() -> None:
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    target = _compose_target()
    loop.update_gaze_context(GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=250), target)

    loop.run_voice_turn("open this", EnvironmentSnapshot(active_app="Mail"))
    events = loop.run_voice_turn("yes", EnvironmentSnapshot(active_app="Mail"))

    assert any(event.result and event.result.status.value == "success" for event in events)
    assert any(
        event.result
        and event.result.details["commands"][0][2] == "interaction.platform.macos_runtime"
        and event.result.details["commands"][0][3] == "double-click-normalized"
        for event in events
    )


def test_fusion_loop_right_click_this_executes_after_confirmation() -> None:
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    target = _compose_target()
    loop.update_gaze_context(GazeObservation(confidence=0.9, x_norm=0.55, y_norm=0.22, fixation_ms=250), target)

    first_events = loop.run_voice_turn("right click this", EnvironmentSnapshot(active_app="Mail"))
    second_events = loop.run_voice_turn("yes", EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == FusionLoopPhase.CONFIRMING for event in first_events)
    assert any(
        event.result
        and event.result.details["commands"][0][2] == "interaction.platform.macos_runtime"
        and event.result.details["commands"][0][3] == "right-click-normalized"
        for event in second_events
    )


def test_fusion_loop_clarifies_when_gaze_is_stale() -> None:
    state = SharedInteractionState(gaze_context_window_ms=300)
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True), state=state)
    target = _compose_target()
    loop.update_gaze_context(GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=100), target)
    state.advance(400)

    events = loop.run_voice_turn("click this", EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == FusionLoopPhase.RECOVERING for event in events)
    assert all(event.result is None for event in events)


def test_fusion_loop_clarifies_when_grounding_confidence_is_too_low() -> None:
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    target = _compose_target(confidence=0.4)
    loop.update_gaze_context(GazeObservation(confidence=0.35, x_norm=0.56, y_norm=0.20, fixation_ms=250), target)

    events = loop.run_voice_turn("open this", EnvironmentSnapshot(active_app="Mail"))

    assert any(event.phase == FusionLoopPhase.RECOVERING for event in events)
    assert any(event.message.startswith("I am not confident enough to act") for event in events)
    assert all(event.result is None for event in events)


def test_fusion_metrics_record_execution_and_confirmation() -> None:
    loop = MultimodalFusionLoop(adapter=MacOSPlatformAdapter(dry_run=True))
    target = _compose_target()
    loop.update_gaze_context(GazeObservation(confidence=0.9, x_norm=0.56, y_norm=0.20, fixation_ms=250), target)

    loop.run_voice_turn("open this", EnvironmentSnapshot(active_app="Mail"))
    loop.run_voice_turn("yes", EnvironmentSnapshot(active_app="Mail"))

    metrics = loop.metrics.to_dict()
    assert metrics["total_turns"] == 2
    assert metrics["confirmations"] >= 1
    assert metrics["executions"] >= 1
