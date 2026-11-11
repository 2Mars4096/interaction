import pytest

from interaction.contracts import EnvironmentSnapshot, GazeObservation
from interaction.runtime import MultimodalFusionLoop, SharedInteractionState, VoiceCommandLoop
from interaction.vision import NormalizedScreenTarget


class ManualClock:
    def __init__(self, now_ms=0):
        self.now_ms = now_ms

    def __call__(self):
        return self.now_ms


def target(target_id="compose"):
    return NormalizedScreenTarget(
        target_id=target_id, label=target_id, role="button",
        x=0.4, y=0.4, width=0.4, height=0.4,
    ).to_grounded_target(confidence=0.95)


def observation(x=0.5, y=0.5):
    return GazeObservation(confidence=0.95, x_norm=x, y_norm=y, fixation_ms=200)


def pending_click(clock):
    state = SharedInteractionState(clock=clock)
    loop = MultimodalFusionLoop(state=state)
    loop.update_gaze_context(observation(), target())
    environment = EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")
    loop.run_voice_turn("click", environment)
    assert loop.pending_decision is not None
    return loop, environment


def test_clock_elapsed_time_ages_gaze_without_recording_more_frames():
    clock = ManualClock()
    state = SharedInteractionState(clock=clock, gaze_context_window_ms=500)
    state.record_gaze(observation(), target())
    assert state.now_ms == 0  # Fixation duration is not elapsed wall time.

    clock.now_ms = 200
    state.record_transcript("click")
    assert state.latest_target_age_ms() == 200

    clock.now_ms = 600
    assert state.now_ms == 600
    assert state.latest_target_age_ms() == 600
    assert state.latest_target() is None
    assert state.candidate_targets() == []


def test_delayed_worker_reading_keeps_its_original_timestamp():
    clock = ManualClock(4_000)
    state = SharedInteractionState(clock=clock)
    state.record_gaze(observation(), target(), observed_at_ms=1_000)

    assert state.latest_target_age_ms() == 3_000
    assert state.latest_target() is None
    state.record_transcript("click")
    assert state.latest_target() is None


def test_out_of_order_and_future_readings_cannot_replace_or_reacquire_a_target():
    clock = ManualClock(4_000)
    state = SharedInteractionState(clock=clock)
    state.record_gaze(observation(), target("new"), observed_at_ms=4_000)
    state.record_gaze(observation(), target("old"), observed_at_ms=3_000)
    state.record_gaze(observation(), target("future"), observed_at_ms=5_000)
    assert state.latest_target().target_id == "new"

    clock.now_ms = 4_500
    state.clear_gaze()
    state.record_gaze(observation(), target("delayed"), observed_at_ms=4_000)
    assert state.latest_target() is None
    assert state.candidate_targets() == []


def test_missing_target_invalidates_previous_history():
    clock = ManualClock()
    state = SharedInteractionState(clock=clock)
    state.record_gaze(observation(), target())
    clock.now_ms = 10
    state.record_gaze(observation(), None)

    assert state.latest_target() is None
    assert state.latest_observation() is None
    assert state.candidate_targets() == []


def test_voice_confirmation_expires_and_weak_retry_does_not_extend_deadline():
    clock = ManualClock()
    loop = VoiceCommandLoop(clock=clock)
    environment = EnvironmentSnapshot(active_app="Notes")
    loop.run_text_turn("type hello", environment)
    clock.now_ms = 14_000
    loop.run_text_turn("yes", environment, final_confidence=0.01)
    assert loop.pending_decision is not None

    clock.now_ms = 15_000
    events = loop.run_text_turn("yes", environment)
    assert loop.pending_decision is None
    assert all(event.result is None for event in events)
    assert any("expired" in (event.message or "") for event in events)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "changed_environment",
    [
        EnvironmentSnapshot(active_app="Terminal", active_window_title="Inbox"),
        EnvironmentSnapshot(active_app="Mail", active_window_title="Compose"),
        EnvironmentSnapshot(),
    ],
)
def test_confirmation_cancels_after_app_window_or_context_loss(fused, changed_environment):
    clock = ManualClock()
    loop = MultimodalFusionLoop(state=SharedInteractionState(clock=clock)) if fused else VoiceCommandLoop(clock=clock)
    run = loop.run_voice_turn if fused else loop.run_text_turn
    run("type hello", EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox"))

    events = run("yes", changed_environment)

    assert loop.pending_decision is None
    assert all(event.result is None for event in events)
    assert any("window changed" in (event.message or "") for event in events)


def test_unknown_dry_run_context_can_confirm_when_it_remains_unknown():
    loop = VoiceCommandLoop(clock=ManualClock())
    loop.run_text_turn("type hello", EnvironmentSnapshot())

    events = loop.run_text_turn("yes", EnvironmentSnapshot())

    assert len([event for event in events if event.result is not None]) == 1


def test_stored_environment_is_a_snapshot_not_a_mutable_reference():
    loop = VoiceCommandLoop(clock=ManualClock())
    environment = EnvironmentSnapshot(active_app="Notes")
    loop.run_text_turn("type hello", environment)
    environment.active_app = "Terminal"

    events = loop.run_text_turn("yes", environment)

    assert all(event.result is None for event in events)


def test_stale_gaze_blocks_pending_click_inside_confirmation_timeout():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    clock.now_ms = 2_000

    events = loop.run_voice_turn("yes", environment)

    assert loop.pending_decision is None
    assert all(event.result is None for event in events)


def test_tracking_loss_cancels_pending_click_before_reacquisition():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    loop.clear_gaze_context()
    clock.now_ms = 100
    loop.update_gaze_context(observation(), target())

    events = loop.run_voice_turn("yes", environment)

    assert loop.pending_decision is None
    assert all(event.result is None for event in events)


def test_worker_revision_change_can_invalidate_pending_then_apply_its_snapshot():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    clock.now_ms = 100
    loop.clear_gaze_context(observed_at_ms=90)
    loop.update_gaze_context(observation(), target("other"), observed_at_ms=90)

    assert loop.state.latest_target().target_id == "other"
    assert loop.state.latest_target_age_ms() == 10
    events = loop.run_voice_turn("yes", environment)
    assert all(event.result is None for event in events)


def test_tracking_loss_does_not_cancel_unrelated_pending_dictation():
    loop = MultimodalFusionLoop(state=SharedInteractionState(clock=ManualClock()))
    environment = EnvironmentSnapshot(active_app="Notes")
    loop.run_voice_turn("type hello", environment)
    loop.clear_gaze_context()

    events = loop.run_voice_turn("yes", environment)

    assert len([event for event in events if event.result is not None]) == 1


def test_changed_target_cancels_even_if_gaze_returns_to_original_target():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    loop.update_gaze_context(observation(), target("other"))
    loop.update_gaze_context(observation(), target())

    events = loop.run_voice_turn("yes", environment)

    assert all(event.result is None for event in events)


def test_large_point_drift_inside_same_screen_region_cancels_click():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    loop.update_gaze_context(observation(x=0.7), target())

    events = loop.run_voice_turn("yes", environment)

    assert all(event.result is None for event in events)
    assert loop.pending_decision is None


def test_fresh_nearby_observation_allows_single_confirmation():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    clock.now_ms = 4_000
    loop.update_gaze_context(observation(x=0.51), target(), observed_at_ms=3_990)

    events = loop.run_voice_turn("yes", environment)

    assert len([event for event in events if event.result is not None]) == 1
    assert loop.pending_decision is None


def test_explicit_simulated_time_is_shared_by_fusion_pending_expiry():
    clock = ManualClock()
    loop, environment = pending_click(clock)
    loop.state.advance(16_000)
    loop.update_gaze_context(observation(), target())

    events = loop.run_voice_turn("yes", environment)

    assert all(event.result is None for event in events)
    assert loop.pending_decision is None


def test_cli_can_clear_pending_through_public_property():
    loop = VoiceCommandLoop(clock=ManualClock())
    environment = EnvironmentSnapshot(active_app="Notes")
    loop.run_text_turn("type hello", environment)
    loop.pending_decision = None

    events = loop.run_text_turn("yes", environment)

    assert all(event.result is None for event in events)
