"""A single camera owner that keeps gaze observations current during speech."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import monotonic
from collections.abc import Callable
import math

from interaction.contracts import GazeObservation, GroundedTarget
from interaction.vision import CalibrationProfile, GazeSample, GazeSmoother, GazeTargetInferencer, NormalizedPoint, WebcamProviderError
from interaction.vision.stability import StableGazeLock

from .clock import monotonic_ms
from .webcam import default_webcam_targets


@dataclass(frozen=True)
class LiveGazeSnapshot:
    observation: GazeObservation
    target: GroundedTarget
    observed_at_ms: int
    revision: int
    sample_count: int = 0
    stable_duration_ms: int = 0


class LiveGazeSampler:
    """Capture on one worker; expose fresh immutable snapshots to the caller.

    This worker never moves the pointer or executes actions. A missing reading
    immediately invalidates its lock. All provider lifecycle calls belong to it.
    """

    def __init__(self, provider, profile: CalibrationProfile, *, clock: Callable[[], int] = monotonic_ms,
                 max_age_ms: int = 1500, interval_s: float = 0.05, min_samples: int = 3,
                 min_stable_ms: int = 120, max_gap_ms: int = 250,
                 max_radius: float = 0.05, min_confidence: float = 0.6) -> None:
        self.provider = provider
        self.profile = profile
        self.clock = clock
        self.max_age_ms = max_age_ms
        self.interval_s = max(0.005, interval_s)
        self._lock = StableGazeLock(
            min_samples=min_samples, min_stable_ms=min_stable_ms, max_gap_ms=max_gap_ms,
            max_radius=max_radius, min_confidence=min_confidence,
        )
        self._condition = Condition()
        self._stop = Event()
        self._thread: Thread | None = None
        self._snapshot: LiveGazeSnapshot | None = None
        self._error: WebcamProviderError | None = None
        self._close_error: WebcamProviderError | None = None
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
        if snapshot is None:
            return None
        now = self.clock()
        if (
            not _finite_number(now)
            or not 0 <= now - snapshot.observed_at_ms <= min(self.max_age_ms, self._lock.max_gap_ms)
        ):
            self._snapshot = None
            self._lock.reset()
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
        try:
            self.provider.open()
            while not self._stop.is_set():
                reading = self.provider.read(delta_ms=1)
                now = self.clock()
                with self._condition:
                    self.attempted_frames += 1
                    if reading is None:
                        self.missing_readings += 1
                        self._lock.reset()
                        self._snapshot = None
                        smoother.reset()
                    else:
                        self.successful_readings += 1
                        observation = self._project_observation(reading.sample)
                        target = inferencer.infer(observation, targets) if observation is not None else None
                        if target is not None:
                            self.grounded_readings += 1
                        previous_revision = self._lock.revision
                        stable = self._lock.update(observation, target, observed_at_ms=now)
                        if self._lock.revision != previous_revision or self._lock.sample_count <= 1:
                            smoother.reset()
                        self._snapshot = None
                        if self._lock.sample_count:
                            # Smoothing is presentation only: it never supplies
                            # evidence to the lock or hides a raw target change.
                            smoothed = smoother.smooth(GazeSample(
                                point=NormalizedPoint(observation.x_norm, observation.y_norm),
                                confidence=observation.confidence, delta_ms=self._lock.duration_ms,
                            ))
                            if stable:
                                self._snapshot = LiveGazeSnapshot(
                                    smoothed, target, now, self._lock.revision,
                                    self._lock.sample_count, self._lock.duration_ms,
                                )
                    self._condition.notify_all()
                self._stop.wait(self.interval_s)
        except Exception as error:
            with self._condition:
                self._error = error if isinstance(error, WebcamProviderError) else WebcamProviderError("camera_worker_failed", str(error))
                self._lock.reset()
                self._snapshot = None
                self._condition.notify_all()
        finally:
            try:
                self.provider.close()
            except Exception as error:
                with self._condition:
                    self._close_error = WebcamProviderError("camera_close_failed", str(error))
                    self._error = self._close_error
                    self._lock.reset()
                    self._snapshot = None
            finally:
                self._stop.set()
                with self._condition:
                    self._lock.reset()
                    self._snapshot = None
                    self._condition.notify_all()

    def _project_observation(self, sample: GazeSample) -> GazeObservation | None:
        if (
            not _finite_number(sample.confidence)
            or not self._lock.min_confidence <= sample.confidence <= 1
            or not all(_finite_number(value) and 0 <= value <= 1 for value in (sample.point.x, sample.point.y))
        ):
            return None
        try:
            point = self.profile.project(sample.point)
        except ValueError:
            return None
        # Admit floating-point roundoff only; actual overshoot must not become
        # artificial stability at a clamped screen edge.
        if not all(_finite_number(value) and -1e-9 <= value <= 1 + 1e-9 for value in (point.x, point.y)):
            return None
        return GazeObservation(
            confidence=sample.confidence,
            x_norm=min(1.0, max(0.0, point.x)), y_norm=min(1.0, max(0.0, point.y)),
            fixation_ms=0,
        )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
