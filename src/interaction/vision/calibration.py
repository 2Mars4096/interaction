"""Calibration fitting and independent quality checks for normalized gaze."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, hypot, isfinite


CALIBRATION_SCHEMA_VERSION = 2
DEFAULT_TRACKER_ID = "opencv-pupil-v1"
# Engineering floors and coarse-target gates, not measured device accuracy.
MIN_RAW_SPAN = 0.005
MAX_MEAN_ERROR = 0.12
MAX_POINT_ERROR = 0.20
MAX_MAPPED_STDDEV = 0.05
MIN_VALIDATION_POINTS = 4
MIN_VALIDATION_SAMPLES = 3


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class CalibrationSample:
    raw: NormalizedPoint
    expected: NormalizedPoint


@dataclass(frozen=True)
class CalibrationValidationPoint:
    raw: NormalizedPoint
    expected: NormalizedPoint
    projected: NormalizedPoint
    error: float
    raw_stddev: NormalizedPoint
    mapped_stddev: NormalizedPoint
    sample_count: int


@dataclass(frozen=True)
class CalibrationQuality:
    points: tuple[CalibrationValidationPoint, ...]
    point_count: int
    sample_count: int
    mean_error: float
    max_error: float
    max_mapped_stddev: float
    mean_error_limit: float
    error_limit: float
    mapped_stddev_limit: float


@dataclass(frozen=True)
class CalibrationProfile:
    x_scale: float = 1.0
    y_scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    schema_version: int = CALIBRATION_SCHEMA_VERSION
    tracker_id: str | None = None
    camera_index: int | None = None
    training_expected: tuple[NormalizedPoint, ...] = ()
    quality: CalibrationQuality | None = None

    def __post_init__(self) -> None:
        for name in ("x_scale", "y_scale", "x_offset", "y_offset"):
            if not _finite_number(getattr(self, name)):
                raise ValueError(f"Calibration {name} must be finite.")
        if type(self.schema_version) is not int or self.schema_version not in {1, CALIBRATION_SCHEMA_VERSION}:
            raise ValueError("Unsupported calibration schema version.")
        if self.tracker_id is not None and (not isinstance(self.tracker_id, str) or not self.tracker_id.strip()):
            raise ValueError("Calibration tracker identity must be a nonempty string.")
        if self.camera_index is not None:
            _check_camera_index(self.camera_index)
        for point in self.training_expected:
            _check_point(point, name="fitting target", normalized=True)
        if self.quality is not None and not isinstance(self.quality, CalibrationQuality):
            raise ValueError("Calibration quality must be typed validation evidence.")

    @classmethod
    def fit(
        cls, samples: list[CalibrationSample], *, min_raw_span: float = MIN_RAW_SPAN,
    ) -> "CalibrationProfile":
        """Fit signed per-axis least squares; validation remains a separate step."""
        if len(samples) < 2:
            raise ValueError("Calibration fitting requires at least two samples.")
        if not _finite_number(min_raw_span) or min_raw_span <= 0:
            raise ValueError("The minimum raw calibration span must be positive and finite.")
        for sample in samples:
            _check_point(sample.raw, name="raw fitting point", normalized=True)
            _check_point(sample.expected, name="fitting target", normalized=True)
        x_scale, x_offset = _fit_axis(
            [sample.raw.x for sample in samples], [sample.expected.x for sample in samples], min_raw_span,
        )
        y_scale, y_offset = _fit_axis(
            [sample.raw.y for sample in samples], [sample.expected.y for sample in samples], min_raw_span,
        )
        return cls(
            x_scale=x_scale, y_scale=y_scale, x_offset=x_offset, y_offset=y_offset,
            training_expected=tuple(sample.expected for sample in samples),
        )

    def project(self, point: NormalizedPoint) -> NormalizedPoint:
        """Map a point without clamping so validation cannot hide overshoot."""
        _check_point(point, name="gaze point")
        projected = NormalizedPoint(
            x=point.x * self.x_scale + self.x_offset,
            y=point.y * self.y_scale + self.y_offset,
        )
        _check_point(projected, name="projected gaze point")
        return projected

    def apply(self, point: NormalizedPoint) -> NormalizedPoint:
        projected = self.project(point)
        return NormalizedPoint(x=_clamp(projected.x), y=_clamp(projected.y))

    def validate(
        self,
        validation_samples: list[CalibrationSample],
        *,
        raw_stddevs: list[NormalizedPoint],
        sample_counts: list[int],
        camera_index: int = 0,
        tracker_id: str = DEFAULT_TRACKER_ID,
        max_mean_error: float = MAX_MEAN_ERROR,
        max_error: float = MAX_POINT_ERROR,
        max_mapped_stddev: float = MAX_MAPPED_STDDEV,
    ) -> "CalibrationProfile":
        """Attach measured held-out quality, without refitting the mapping."""
        _check_camera_index(camera_index)
        if not isinstance(tracker_id, str) or not tracker_id.strip():
            raise ValueError("Validation requires a tracker identity.")
        if self.schema_version != CALIBRATION_SCHEMA_VERSION or len(self.training_expected) < 2:
            raise ValueError("Fit the calibration before collecting independent validation points.")
        for limit in (max_mean_error, max_error, max_mapped_stddev):
            if not _finite_number(limit) or limit <= 0:
                raise ValueError("Validation error and dispersion limits must be positive and finite.")
        if len(validation_samples) < MIN_VALIDATION_POINTS:
            raise ValueError("Calibration validation requires at least four distinct held-out targets.")
        if len(validation_samples) != len(raw_stddevs) or len(validation_samples) != len(sample_counts):
            raise ValueError("Every validation point requires its own dispersion and sample count.")

        points: list[CalibrationValidationPoint] = []
        for sample, raw_stddev, sample_count in zip(validation_samples, raw_stddevs, sample_counts):
            _check_point(sample.raw, name="raw validation point", normalized=True)
            _check_point(sample.expected, name="validation target", normalized=True)
            _check_point(raw_stddev, name="raw validation dispersion", nonnegative=True)
            if type(sample_count) is not int or sample_count < MIN_VALIDATION_SAMPLES:
                raise ValueError("Every validation target requires at least three measured samples.")
            if any(_same_point(sample.expected, expected) for expected in self.training_expected):
                raise ValueError("Validation targets must be independent of fitting targets.")
            if any(_same_point(sample.expected, point.expected) for point in points):
                raise ValueError("Validation targets must be distinct.")
            projected = self.project(sample.raw)
            mapped_stddev = NormalizedPoint(
                x=abs(self.x_scale) * raw_stddev.x,
                y=abs(self.y_scale) * raw_stddev.y,
            )
            _check_point(mapped_stddev, name="mapped validation dispersion", nonnegative=True)
            points.append(CalibrationValidationPoint(
                raw=sample.raw, expected=sample.expected, projected=projected,
                error=hypot(projected.x - sample.expected.x, projected.y - sample.expected.y),
                raw_stddev=raw_stddev, mapped_stddev=mapped_stddev, sample_count=sample_count,
            ))
        quality = CalibrationQuality(
            points=tuple(points), point_count=len(points), sample_count=sum(point.sample_count for point in points),
            mean_error=fsum(point.error for point in points) / len(points),
            max_error=max(point.error for point in points),
            max_mapped_stddev=max(max(point.mapped_stddev.x, point.mapped_stddev.y) for point in points),
            mean_error_limit=max_mean_error, error_limit=max_error, mapped_stddev_limit=max_mapped_stddev,
        )
        if quality.mean_error > max_mean_error or quality.max_error > max_error:
            raise ValueError(
                f"Independent calibration error is too large: mean {quality.mean_error:.3f} "
                f"(limit {max_mean_error:.3f}), maximum {quality.max_error:.3f} (limit {max_error:.3f})."
            )
        if quality.max_mapped_stddev > max_mapped_stddev:
            raise ValueError(
                f"Mapped calibration dispersion is too large: {quality.max_mapped_stddev:.3f} "
                f"(limit {max_mapped_stddev:.3f})."
            )
        return replace(self, quality=quality, camera_index=camera_index, tracker_id=tracker_id)

    def is_usable(self, *, camera_index: int | None = None, tracker_id: str = DEFAULT_TRACKER_ID) -> bool:
        """Recheck persisted evidence and current acceptance gates for live use."""
        if (
            self.schema_version != CALIBRATION_SCHEMA_VERSION
            or self.quality is None
            or self.camera_index is None
            or self.tracker_id != tracker_id
        ):
            return False
        if camera_index is not None and (type(camera_index) is not int or camera_index != self.camera_index):
            return False
        quality = self.quality
        try:
            checked = self.validate(
                [CalibrationSample(point.raw, point.expected) for point in quality.points],
                raw_stddevs=[point.raw_stddev for point in quality.points],
                sample_counts=[point.sample_count for point in quality.points],
                camera_index=self.camera_index, tracker_id=tracker_id,
                max_mean_error=quality.mean_error_limit, max_error=quality.error_limit,
                max_mapped_stddev=quality.mapped_stddev_limit,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return (
            checked.quality == quality
            and quality.mean_error <= MAX_MEAN_ERROR
            and quality.max_error <= MAX_POINT_ERROR
            and quality.max_mapped_stddev <= MAX_MAPPED_STDDEV
        )


def _fit_axis(raw_values: list[float], expected_values: list[float], min_raw_span: float) -> tuple[float, float]:
    raw_span = max(raw_values) - min(raw_values)
    if raw_span < min_raw_span:
        raise ValueError(f"Raw calibration movement is insufficient ({raw_span:.6f}; minimum {min_raw_span:.6f}).")
    if max(expected_values) == min(expected_values):
        raise ValueError("Fitting targets must span both screen axes.")
    raw_mean = fsum(raw_values) / len(raw_values)
    expected_mean = fsum(expected_values) / len(expected_values)
    variance = fsum((raw - raw_mean) ** 2 for raw in raw_values)
    if variance <= 0:
        raise ValueError("Raw calibration samples must vary on both axes.")
    scale = fsum((raw - raw_mean) * (expected - expected_mean) for raw, expected in zip(raw_values, expected_values)) / variance
    offset = expected_mean - raw_mean * scale
    if not _finite_number(scale) or not _finite_number(offset) or scale == 0:
        raise ValueError("Calibration samples do not define a finite, responsive mapping.")
    return scale, offset


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _check_point(point: NormalizedPoint, *, name: str, normalized: bool = False, nonnegative: bool = False) -> None:
    if not isinstance(point, NormalizedPoint) or not all(_finite_number(value) for value in (point.x, point.y)):
        raise ValueError(f"The {name} must contain finite numeric coordinates.")
    if normalized and not (0 <= point.x <= 1 and 0 <= point.y <= 1):
        raise ValueError(f"The {name} must be inside the normalized screen or camera bounds.")
    if nonnegative and (point.x < 0 or point.y < 0):
        raise ValueError(f"The {name} must be nonnegative.")


def _check_camera_index(camera_index: int) -> None:
    if type(camera_index) is not int or camera_index < 0:
        raise ValueError("Calibration camera index must be a nonnegative integer.")


def _same_point(left: NormalizedPoint, right: NormalizedPoint) -> bool:
    return hypot(left.x - right.x, left.y - right.y) <= 1e-9


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
