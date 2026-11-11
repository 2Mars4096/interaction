from queue import Empty, Queue
from threading import get_ident
from types import SimpleNamespace

import pytest

from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint, WebcamProviderError


def reading(x=0.15, y=0.15, confidence=0.95):
    return SimpleNamespace(sample=GazeSample(NormalizedPoint(x, y), confidence=confidence, delta_ms=100))


class Provider:
    def __init__(self):
        self.queue = Queue()
        self.owner = None
        self.closed = False

    def open(self):
        self.owner = get_ident()

    def read(self, **kwargs):
        assert self.owner == get_ident()
        try:
            return self.queue.get(timeout=0.2)
        except Empty:
            return None

    def close(self):
        assert self.owner == get_ident()
        self.closed = True


def test_worker_owns_lifecycle_and_continues_after_initial_lock():
    provider = Provider()
    provider.queue.put(reading())
    provider.queue.put(reading())
    sampler = LiveGazeSampler(provider, CalibrationProfile(), interval_s=0.005, min_stable_ms=0, min_samples=2)
    sampler.start()
    try:
        assert sampler.wait_for_target(timeout_s=1) is not None
        assert provider.owner != get_ident()
        provider.queue.put(reading())
        with sampler._condition:
            assert sampler._condition.wait_for(lambda: sampler.successful_readings >= 3, timeout=1)
        assert sampler.latest().target.target_id == "upper_left"
    finally:
        sampler.stop()
    assert provider.closed
    assert not sampler._thread.is_alive()
    assert sampler.latest() is None


def test_loss_invalidates_lock_and_requires_new_samples():
    provider = Provider()
    provider.queue.put(reading())
    provider.queue.put(reading())
    sampler = LiveGazeSampler(provider, CalibrationProfile(), interval_s=0.005, min_stable_ms=0, min_samples=2)
    sampler.start()
    try:
        original = sampler.wait_for_target(timeout_s=1)
        assert original
        provider.queue.put(None)
        provider.queue.put(reading())
        with sampler._condition:
            assert sampler._condition.wait_for(lambda: sampler.attempted_frames >= 4, timeout=1)
        assert sampler.latest() is None
        provider.queue.put(reading())
        assert sampler.wait_for_target(timeout_s=1).revision > original.revision
    finally:
        sampler.stop()


def test_snapshot_expires_using_real_clock_domain():
    now = [0]
    provider = Provider()
    provider.queue.put(reading())
    provider.queue.put(reading())
    sampler = LiveGazeSampler(provider, CalibrationProfile(), clock=lambda: now[0], min_stable_ms=0, min_samples=2, interval_s=0.005)
    sampler.start()
    try:
        assert sampler.wait_for_target(timeout_s=1)
        now[0] = 1600
        assert sampler.latest() is None
    finally:
        sampler.stop()


def test_worker_open_failure_is_structured_and_closed():
    class FailingProvider(Provider):
        def open(self):
            super().open()
            raise WebcamProviderError("camera_unavailable", "Camera unavailable")

    provider = FailingProvider()
    sampler = LiveGazeSampler(provider, CalibrationProfile())
    sampler.start()
    with pytest.raises(WebcamProviderError, match="Camera unavailable"):
        sampler.wait_for_target(timeout_s=1)
    sampler.stop()
    assert provider.closed


def test_worker_read_failure_is_reported_and_camera_is_closed():
    class BrokenReadProvider(Provider):
        def read(self, **kwargs):
            raise OSError("Camera disconnected")

    provider = BrokenReadProvider()
    sampler = LiveGazeSampler(provider, CalibrationProfile())
    sampler.start()
    with pytest.raises(WebcamProviderError, match="Camera disconnected") as error:
        sampler.wait_for_target(timeout_s=1)
    assert error.value.code == "camera_worker_failed"
    sampler.stop()
    assert provider.closed


def test_worker_close_failure_is_reported_to_caller():
    class BrokenCloseProvider(Provider):
        def close(self):
            super().close()
            raise OSError("Camera release failed")

    provider = BrokenCloseProvider()
    provider.queue.put(reading())
    provider.queue.put(reading())
    sampler = LiveGazeSampler(provider, CalibrationProfile(), interval_s=0.005, min_stable_ms=0, min_samples=2)
    sampler.start()
    assert sampler.wait_for_target(timeout_s=1)
    with pytest.raises(WebcamProviderError, match="Camera release failed") as error:
        sampler.stop()
    assert error.value.code == "camera_close_failed"
    assert not sampler._thread.is_alive()


class TimedProvider(Provider):
    def __init__(self):
        super().__init__()
        self.now = 0

    def emit(self, at_ms, sample):
        self.queue.put((at_ms, sample))

    def read(self, **kwargs):
        item = super().read(**kwargs)
        if item is None:
            return None
        self.now, sample = item
        return sample


def wait_for_frames(sampler, count):
    with sampler._condition:
        assert sampler._condition.wait_for(lambda: sampler.attempted_frames >= count, timeout=1)


def timed_sampler(provider, *, profile=None):
    return LiveGazeSampler(
        provider, profile or CalibrationProfile(), clock=lambda: provider.now, interval_s=0.005,
    )


@pytest.mark.parametrize("invalid", [None, reading(confidence=0.1), reading(x=float("nan")), reading(y=1.1)])
def test_worker_invalid_evidence_resets_lock_and_smoothing_before_recovery(invalid):
    provider = TimedProvider()
    for at_ms in (0, 60, 120):
        provider.emit(at_ms, reading())
    sampler = timed_sampler(provider)
    sampler.start()
    try:
        original = sampler.wait_for_target(timeout_s=1)
        assert original is not None
        assert original.sample_count == 3 and original.stable_duration_ms == 120
        provider.emit(180, invalid)
        provider.emit(240, reading(x=0.28))
        wait_for_frames(sampler, 5)
        assert sampler.latest() is None
        provider.emit(300, reading(x=0.28))
        wait_for_frames(sampler, 6)
        assert sampler.latest() is None
        provider.emit(360, reading(x=0.28))
        recovered = sampler.wait_for_target(timeout_s=1)
        assert recovered is not None and recovered.revision > original.revision
        assert recovered.observation.x_norm == pytest.approx(0.28)
        assert recovered.sample_count == 3 and recovered.stable_duration_ms == 120
    finally:
        sampler.stop()


def test_worker_jitter_in_one_region_cannot_be_hidden_by_smoothing():
    provider = TimedProvider()
    for index, x in enumerate([0.10, 0.25] * 4):
        provider.emit(index * 60, reading(x=x))
    sampler = timed_sampler(provider)
    sampler.start()
    try:
        assert sampler.wait_for_target(max_frames=8, timeout_s=1) is None
        assert sampler.grounded_readings == 8
    finally:
        sampler.stop()


def test_worker_projects_without_clamping_overshoot_into_a_stable_edge():
    provider = TimedProvider()
    for index, x in enumerate([0.8, 0.9, 0.7, 0.85]):
        provider.emit(index * 60, reading(x=x))
    sampler = timed_sampler(provider, profile=CalibrationProfile(x_scale=2.0))
    sampler.start()
    try:
        assert sampler.wait_for_target(max_frames=4, timeout_s=1) is None
        assert sampler.grounded_readings == 0
    finally:
        sampler.stop()


def test_worker_long_read_gap_discards_partial_sample_count():
    provider = TimedProvider()
    for at_ms in (0, 60, 500):
        provider.emit(at_ms, reading())
    sampler = timed_sampler(provider)
    sampler.start()
    try:
        wait_for_frames(sampler, 3)
        assert sampler.latest() is None
        provider.emit(560, reading())
        wait_for_frames(sampler, 4)
        assert sampler.latest() is None
        provider.emit(620, reading())
        snapshot = sampler.wait_for_target(timeout_s=1)
        assert snapshot is not None
        assert snapshot.sample_count == 3 and snapshot.stable_duration_ms == 120
    finally:
        sampler.stop()


def test_worker_snapshot_expires_at_sampling_gap_and_cannot_reappear():
    provider = TimedProvider()
    for at_ms in (0, 60, 120):
        provider.emit(at_ms, reading())
    sampler = timed_sampler(provider)
    sampler.start()
    try:
        original = sampler.wait_for_target(timeout_s=1)
        assert original is not None
        provider.now = 400
        assert sampler.latest() is None
        # Moving the clock back cannot restore a snapshot already invalidated.
        provider.now = 120
        assert sampler.latest() is None
        for at_ms in (460, 520, 580):
            provider.emit(at_ms, reading())
        recovered = sampler.wait_for_target(timeout_s=1)
        assert recovered is not None and recovered.revision > original.revision
    finally:
        sampler.stop()


@pytest.mark.parametrize("bad_timestamp", [float("nan"), float("inf"), -1, 100])
def test_worker_bad_clock_sample_invalidates_without_terminating_camera(bad_timestamp):
    provider = TimedProvider()
    for at_ms in (0, 60, 120):
        provider.emit(at_ms, reading())
    sampler = timed_sampler(provider)
    sampler.start()
    try:
        original = sampler.wait_for_target(timeout_s=1)
        assert original is not None
        provider.emit(bad_timestamp, reading())
        wait_for_frames(sampler, 4)
        assert sampler.latest() is None
        for at_ms in (240, 300, 360):
            provider.emit(at_ms, reading())
        recovered = sampler.wait_for_target(timeout_s=1)
        assert recovered is not None and recovered.revision > original.revision
    finally:
        sampler.stop()
