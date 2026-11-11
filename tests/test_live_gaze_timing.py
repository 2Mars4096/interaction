"""Live gaze decisions use current samples and measured time, never nominal FPS."""

from types import SimpleNamespace

import pytest

from interaction.contracts import ActionName
from interaction.feedback import GazeLoopPhase
from interaction.platform import MacOSPlatformAdapter
from interaction.runtime.gaze import GazeTrackingLoop
from interaction.runtime.sampling import timed_readings
from interaction.runtime.webcam import capture_live_gaze_context, run_live_cursor_follow, run_live_webcam_trace
from interaction.vision import CalibrationProfile, DwellTrigger, GazeSample, NormalizedPoint


class Clock:
    def __init__(self, *, advances=True):
        self.now = 0
        self.advances = advances
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.advances:
            self.now += round(seconds * 1000)


def reading(x=0.15, y=0.15, *, confidence=0.95):
    return SimpleNamespace(sample=GazeSample(NormalizedPoint(x, y), confidence=confidence, delta_ms=10000))


class Provider:
    def __init__(self, clock, readings, *, stalls=None):
        self.clock = clock
        self.readings = list(readings)
        self.stalls = list(stalls or [0] * len(self.readings))
        self.requested_deltas = []

    def read(self, *, delta_ms):
        self.requested_deltas.append(delta_ms)
        self.clock.now += self.stalls.pop(0)
        return self.readings.pop(0)


class RecordingAdapter(MacOSPlatformAdapter):
    def __init__(self, clock):
        super().__init__(dry_run=True)
        self.clock = clock
        self.executed_at = []

    def execute(self, request):
        self.executed_at.append(self.clock())
        return super().execute(request)


def capture(readings, *, stalls=None, interval_ms=100):
    clock = Clock()
    provider = Provider(clock, readings, stalls=stalls)
    return capture_live_gaze_context(
        provider, CalibrationProfile(), frames=len(readings), delta_ms=interval_ms,
        clock=clock, sleep=clock.sleep,
    )


def test_sampling_paces_actual_reads_and_reports_elapsed_time_including_stalls():
    clock = Clock()
    provider = Provider(clock, [reading()] * 3, stalls=[0, 25, 300])
    samples = list(timed_readings(provider, frames=3, interval_ms=100, clock=clock, sleep=clock.sleep))

    assert [sample.observed_at_ms for sample in samples] == [0, 125, 525]
    assert [sample.elapsed_ms for sample in samples] == [0, 125, 400]
    assert all(sample.clock_continuous for sample in samples)
    assert provider.requested_deltas == [1, 1, 1]
    assert clock.sleeps == [0.1, 0.1]


def test_bounded_capture_cannot_return_one_valid_sample_followed_by_missing_frames():
    context, events, summary = capture([reading(), None, None])

    assert context is None
    assert summary["successful_readings"] == 1 and summary["missing_readings"] == 2
    assert events[-1].phase == GazeLoopPhase.RECOVERING


@pytest.mark.parametrize("tail", [reading(confidence=0.1), reading(x=float("nan")), None, reading(x=0.55)])
def test_bounded_capture_discards_previously_stable_target_when_final_evidence_changes(tail):
    context, events, summary = capture([reading(), reading(), reading(), tail])

    assert context is None
    assert summary["attempted_frames"] == 4
    assert any(event.message == "Stable live gaze target acquired." for event in events)


def test_bounded_capture_returns_current_timestamp_and_measured_lock_evidence():
    context, _events, summary = capture([reading(), reading(), reading()])

    assert context is not None
    assert context.observed_at_ms == 200
    assert context.stable_duration_ms == 200
    assert context.dominant_target_hits == 3
    assert context.target.confidence == pytest.approx(0.95)
    assert summary["stable_duration_ms"] == 200


def test_bounded_capture_stall_discards_an_earlier_stable_sequence():
    context, _events, summary = capture([reading()] * 4, stalls=[0, 0, 0, 300])

    assert context is None
    assert summary["successful_readings"] == 4


@pytest.mark.parametrize("frames,interval_ms,expected_moves", [(2, 200, 0), (3, 10, 0), (3, 100, 1)])
def test_cursor_requires_both_sample_count_and_real_stable_duration(frames, interval_ms, expected_moves):
    clock = Clock()
    adapter = RecordingAdapter(clock)
    provider = Provider(clock, [reading()] * frames)
    _events, summary = run_live_cursor_follow(
        provider, CalibrationProfile(), adapter=adapter, frames=frames, delta_ms=interval_ms,
        clock=clock, sleep=clock.sleep,
    )

    assert summary["moved_events"] == expected_moves
    assert len(adapter.executed_at) == expected_moves
    if expected_moves:
        assert adapter.executed_at == [200]


def test_cursor_loss_requires_a_fresh_stable_sequence_before_moving_again():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    samples = [reading()] * 3 + [None] + [reading(x=0.75)] * 3
    provider = Provider(clock, samples)
    _events, summary = run_live_cursor_follow(
        provider, CalibrationProfile(), adapter=adapter, frames=len(samples), delta_ms=100,
        clock=clock, sleep=clock.sleep,
    )

    assert adapter.executed_at == [200, 600]
    assert summary["missing_readings"] == 1
    assert summary["acquiring_readings"] == 4


def test_cursor_cannot_turn_projected_overshoot_into_stable_clamped_edge():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    provider = Provider(clock, [reading(x=x) for x in (0.8, 0.9, 0.75)])
    _events, summary = run_live_cursor_follow(
        provider, CalibrationProfile(x_scale=2), adapter=adapter, frames=3, delta_ms=100,
        clock=clock, sleep=clock.sleep,
    )

    assert adapter.executed_at == []
    assert summary["moved_events"] == 0
    assert summary["low_confidence_readings"] == 3


def test_live_trace_cannot_credit_nominal_sample_duration_when_clock_does_not_advance():
    clock = Clock(advances=False)
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(adapter=adapter, dwell_trigger=DwellTrigger(dwell_ms=200))
    provider = Provider(clock, [reading()] * 8)
    _events, summary = run_live_webcam_trace(
        provider, loop, frames=8, delta_ms=100, clock=clock, sleep=clock.sleep,
    )

    assert clock.now == 0
    assert loop.dwell_trigger.accumulated_ms == 0
    assert summary["triggered_events"] == 0
    assert adapter.executed_at == []


def test_long_camera_read_gap_resets_partial_dwell_before_reacquisition():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(adapter=adapter, dwell_trigger=DwellTrigger(dwell_ms=200))
    provider = Provider(clock, [reading()] * 5, stalls=[0, 0, 300, 0, 0])
    events, summary = run_live_webcam_trace(
        provider, loop, frames=5, delta_ms=100, clock=clock, sleep=clock.sleep,
    )

    assert adapter.executed_at == [700]
    assert summary["triggered_events"] == 1
    assert any(event.phase == GazeLoopPhase.RECOVERING and "sampling paused" in event.message for event in events)


def test_missing_live_read_cancels_armed_drag_before_destination_can_execute():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(
        adapter=adapter,
        dwell_trigger=DwellTrigger(dwell_ms=200, action=ActionName.DRAG_TARGET),
        auto_confirm_actions={ActionName.DRAG_TARGET},
    )
    samples = [reading()] * 3 + [None] + [reading(x=0.75)] * 4
    provider = Provider(clock, samples)
    events, summary = run_live_webcam_trace(
        provider, loop, frames=len(samples), delta_ms=100, clock=clock, sleep=clock.sleep,
    )

    assert any("Drag origin armed" in event.message for event in events)
    assert any("Drag origin" in event.message and "cancelled" in event.message for event in events)
    assert summary["missing_readings"] == 1
    assert loop.pending_drag_origin is None
    assert adapter.executed_at == []
    assert all(event.result is None for event in events)


def test_one_different_target_sample_cannot_credit_unobserved_lookaway_time():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(
        adapter=adapter, dwell_trigger=DwellTrigger(dwell_ms=400, cooldown_ms=0, rearm_ms=200),
    )
    samples = [reading()] * 3 + [reading(x=0.75)] + [reading()] * 3
    provider = Provider(clock, samples)
    _events, summary = run_live_webcam_trace(
        provider, loop, frames=len(samples), delta_ms=200, clock=clock, sleep=clock.sleep,
    )

    assert adapter.executed_at == [400]
    assert summary["triggered_events"] == 1


def test_live_session_completion_cancels_armed_drag_before_loop_reuse():
    clock = Clock()
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(
        adapter=adapter,
        dwell_trigger=DwellTrigger(dwell_ms=200, action=ActionName.DRAG_TARGET),
        auto_confirm_actions={ActionName.DRAG_TARGET},
    )
    events, _summary = run_live_webcam_trace(
        Provider(clock, [reading()] * 3), loop,
        frames=3, delta_ms=100, clock=clock, sleep=clock.sleep,
    )

    assert any("Drag origin armed" in event.message for event in events)
    assert any("Drag origin" in event.message and "cancelled" in event.message for event in events)
    assert loop.pending_drag_origin is None
    assert loop.pending_drag_elapsed_ms == 0
    assert loop.dwell_trigger.accumulated_ms == 0
    assert not loop.smoother.window

    # The following bounded session may acquire a new origin, but cannot finish
    # a drag whose source belonged to the already completed session.
    run_live_webcam_trace(
        Provider(clock, [reading(x=0.75)] * 6), loop,
        frames=6, delta_ms=100, clock=clock, sleep=clock.sleep,
    )
    assert adapter.executed_at == []
    assert loop.pending_drag_origin is None


@pytest.mark.parametrize("failure", [OSError("Camera disconnected"), KeyboardInterrupt()])
def test_live_session_error_or_interruption_clears_armed_drag(failure):
    clock = Clock()
    adapter = RecordingAdapter(clock)
    loop = GazeTrackingLoop(
        adapter=adapter,
        dwell_trigger=DwellTrigger(dwell_ms=200, action=ActionName.DRAG_TARGET),
        auto_confirm_actions={ActionName.DRAG_TARGET},
    )

    class FailingProvider(Provider):
        def read(self, *, delta_ms):
            if not self.readings:
                assert loop.pending_drag_origin is not None
                raise failure
            return super().read(delta_ms=delta_ms)

    with pytest.raises(type(failure)):
        run_live_webcam_trace(
            FailingProvider(clock, [reading()] * 3), loop,
            frames=4, delta_ms=100, clock=clock, sleep=clock.sleep,
        )

    assert loop.pending_drag_origin is None
    assert loop.pending_drag_elapsed_ms == 0
    assert loop.dwell_trigger.accumulated_ms == 0
    assert not loop.smoother.window
    assert adapter.executed_at == []
