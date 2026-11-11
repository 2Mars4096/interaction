"""A single camera owner that keeps gaze observations current during speech."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import monotonic
from collections.abc import Callable

from interaction.contracts import GazeObservation, GroundedTarget
from interaction.vision import CalibrationProfile, GazeSample, GazeSmoother, GazeTargetInferencer, WebcamProviderError

from .clock import monotonic_ms
from .webcam import default_webcam_targets


@dataclass(frozen=True)
class LiveGazeSnapshot:
    observation: GazeObservation
    target: GroundedTarget
    observed_at_ms: int
    revision: int


class LiveGazeSampler:
    """Capture on one worker; expose fresh immutable snapshots to the caller.

    This worker never moves the pointer or executes actions. A missing reading
    immediately invalidates its lock. All provider lifecycle calls belong to it.
    """

    def __init__(self, provider, profile: CalibrationProfile, *, clock: Callable[[], int] = monotonic_ms,
                 max_age_ms: int = 1500, interval_s: float = 0.05, min_samples: int = 3,
                 min_stable_ms: int = 120) -> None:
        self.provider = provider
        self.profile = profile
        self.clock = clock
        self.max_age_ms = max_age_ms
        self.interval_s = max(0.005, interval_s)
        self.min_samples = max(2, min_samples)
        self.min_stable_ms = max(0, min_stable_ms)
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._snapshot: LiveGazeSnapshot | None = None
        self._error: WebcamProviderError | None = None
        self._close_error: WebcamProviderError | None = None
        self._revision = 0
        self.attempted_frames = 0
        self.successful_readings = 0
        self.missing_readings = 0
        self.grounded_readings = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Gaze sampler is already started.")
        self._thread = Thread(target=self._run, name="interaction-gaze", daemon=True)
        self._thread.start()

    def wait_for_target(self, *, max_frames: int = 12, timeout_s: float = 5.0) -> LiveGazeSnapshot | None:
        deadline = monotonic() + timeout_s
        with self._condition:
            while self._snapshot is None and self._error is None and self.attempted_frames < max_frames:
                remaining = deadline - monotonic()
                if remaining <= 0 or self._stop.is_set():
                    break
                self._condition.wait(remaining)
            return self._current()

    def latest(self) -> LiveGazeSnapshot | None:
        with self._condition:
            return self._current()

    def _current(self) -> LiveGazeSnapshot | None:
        if self._error is not None:
            raise self._error
        snapshot = self._snapshot
        if snapshot is None or not 0 <= self.clock() - snapshot.observed_at_ms <= self.max_age_ms:
            return None
        return snapshot

    def summary(self) -> dict[str, int]:
        with self._condition:
            return {"attempted_frames": self.attempted_frames, "successful_readings": self.successful_readings,
                    "missing_readings": self.missing_readings, "grounded_readings": self.grounded_readings}

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise WebcamProviderError("camera_shutdown_timeout", "The camera did not finish shutting down. Close this session before starting another.")
        if self._close_error is not None:
            raise self._close_error

    def _run(self) -> None:
        smoother = GazeSmoother()
        inferencer = GazeTargetInferencer()
        targets = default_webcam_targets()
        candidate_id = None
        candidate_count = 0
        candidate_since = self.clock()
        previous_time = self.clock()
        try:
            self.provider.open()
            while not self._stop.is_set():
                reading = self.provider.read(delta_ms=1)
                now = self.clock()
                elapsed = max(1, now - previous_time)
                previous_time = now
                with self._condition:
                    self.attempted_frames += 1
                    if reading is None:
                        self.missing_readings += 1
                        if candidate_id is not None:
                            self._revision += 1
                        self._snapshot = None
                        candidate_id = None
                        candidate_count = 0
                        smoother = GazeSmoother()
                    else:
                        self.successful_readings += 1
                        observation = smoother.smooth(GazeSample(
                            point=self.profile.apply(reading.sample.point), confidence=reading.sample.confidence, delta_ms=elapsed,
                        ))
                        target = inferencer.infer(observation, targets)
                        if target is None:
                            if candidate_id is not None:
                                self._revision += 1
                            self._snapshot = None
                            candidate_id = None
                            candidate_count = 0
                        else:
                            self.grounded_readings += 1
                            if candidate_id != target.target_id:
                                self._revision += 1
                                candidate_id = target.target_id
                                candidate_count = 0
                                candidate_since = now
                                self._snapshot = None
                            candidate_count += 1
                            if candidate_count >= self.min_samples and now - candidate_since >= self.min_stable_ms:
                                self._snapshot = LiveGazeSnapshot(observation, target, now, self._revision)
                    self._condition.notify_all()
                self._stop.wait(self.interval_s)
        except Exception as error:
            with self._condition:
                self._error = error if isinstance(error, WebcamProviderError) else WebcamProviderError("camera_worker_failed", str(error))
                self._snapshot = None
                self._condition.notify_all()
        finally:
            try:
                self.provider.close()
            except Exception as error:
                with self._condition:
                    self._close_error = WebcamProviderError("camera_close_failed", str(error))
                    self._error = self._close_error
                    self._snapshot = None
            finally:
                self._stop.set()
                with self._condition:
                    self._condition.notify_all()
