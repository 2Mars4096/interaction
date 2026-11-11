from queue import Empty, Queue
from threading import get_ident
from types import SimpleNamespace

import pytest

from interaction.runtime.live_gaze import LiveGazeSampler
from interaction.vision import CalibrationProfile, GazeSample, NormalizedPoint, WebcamProviderError


def reading(x=0.15, y=0.15):
    return SimpleNamespace(sample=GazeSample(NormalizedPoint(x, y), confidence=0.95, delta_ms=100))


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
