from dataclasses import replace

import pytest

from interaction.vision import CalibrationProfile, CalibrationSample, NormalizedPoint


FIT_POINTS = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]
VALIDATION_POINTS = [(0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)]


def samples(points=FIT_POINTS, raw_transform=lambda point: point):
    return [CalibrationSample(raw_transform(NormalizedPoint(*point)), NormalizedPoint(*point)) for point in points]


def validate(profile, validation_samples=None, **kwargs):
    validation_samples = samples(VALIDATION_POINTS) if validation_samples is None else validation_samples
    kwargs.setdefault("raw_stddevs", [NormalizedPoint(0.001, 0.001)] * len(validation_samples))
    kwargs.setdefault("sample_counts", [6] * len(validation_samples))
    return profile.validate(validation_samples, **kwargs)


def test_signed_fit_handles_reversed_axes_and_stays_unvalidated():
    profile = CalibrationProfile.fit(samples(raw_transform=lambda point: NormalizedPoint(1 - point.x, 0.2 + point.y * 0.6)))

    assert profile.x_scale == pytest.approx(-1)
    assert profile.y_scale == pytest.approx(1 / 0.6)
    projected = profile.project(NormalizedPoint(0.65, 0.59))
    assert projected.x == pytest.approx(0.35)
    assert projected.y == pytest.approx(0.65)
    assert profile.training_expected == tuple(NormalizedPoint(*point) for point in FIT_POINTS)
    assert profile.quality is None
    assert not profile.is_usable()


def test_fit_uses_all_pairs_in_least_squares_instead_of_extreme_ranges():
    raw_values = [0.0, 0.2, 0.8, 1.0]
    expected_values = [0.0, 0.8, 0.2, 1.0]
    profile = CalibrationProfile.fit([
        CalibrationSample(NormalizedPoint(raw, raw), NormalizedPoint(expected, expected))
        for raw, expected in zip(raw_values, expected_values)
    ])

    assert profile.x_scale == pytest.approx(8 / 17)
    assert profile.x_offset == pytest.approx(9 / 34)
    assert profile.y_scale == pytest.approx(8 / 17)


@pytest.mark.parametrize("fitting_samples", [[], samples()[:1]])
def test_fit_rejects_insufficient_input(fitting_samples):
    with pytest.raises(ValueError, match="at least two"):
        CalibrationProfile.fit(fitting_samples)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("span", [0.0, 0.004])
def test_fit_rejects_constant_or_too_small_raw_axis(axis, span):
    def raw_point(point):
        return replace(point, **{axis: 0.5 + getattr(point, axis) * span})

    with pytest.raises(ValueError, match="movement is insufficient"):
        CalibrationProfile.fit(samples(raw_transform=raw_point))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), "0.5", True])
@pytest.mark.parametrize("field", ["raw", "expected"])
def test_fit_rejects_nonfinite_or_nonnumeric_points(field, invalid):
    fitting_samples = samples()
    fitting_samples[0] = replace(fitting_samples[0], **{field: NormalizedPoint(invalid, 0.2)})
    with pytest.raises(ValueError):
        CalibrationProfile.fit(fitting_samples)


def test_project_preserves_overshoot_while_apply_clamps_output():
    profile = CalibrationProfile(x_scale=2, y_scale=-2, x_offset=-0.5, y_offset=1.5)

    assert profile.project(NormalizedPoint(0.9, 0.9)) == NormalizedPoint(1.3, -0.30000000000000004)
    assert profile.apply(NormalizedPoint(0.9, 0.9)) == NormalizedPoint(1, 0)


def test_validation_records_independent_errors_counts_dispersion_and_identity_without_refitting():
    fitted = CalibrationProfile.fit(samples())
    validated = validate(fitted, camera_index=2)

    assert fitted.quality is None
    assert (validated.x_scale, validated.y_scale, validated.x_offset, validated.y_offset) == (
        fitted.x_scale, fitted.y_scale, fitted.x_offset, fitted.y_offset
    )
    assert validated.schema_version == 2
    assert validated.tracker_id == "opencv-pupil-v1"
    assert validated.camera_index == 2
    assert validated.quality.point_count == 4
    assert validated.quality.sample_count == 24
    assert validated.quality.mean_error == pytest.approx(0)
    assert validated.quality.max_error == pytest.approx(0)
    assert validated.quality.max_mapped_stddev == pytest.approx(0.001)
    assert all(point.sample_count == 6 for point in validated.quality.points)
    assert validated.is_usable(camera_index=2)
    assert not validated.is_usable(camera_index=0)
    assert not validated.is_usable(camera_index=True)
    assert not validated.is_usable(camera_index=2, tracker_id="another-tracker")


@pytest.mark.parametrize("validation_samples", [samples(VALIDATION_POINTS)[:3], samples(VALIDATION_POINTS)[:1] * 4, samples()[:4]])
def test_validation_rejects_few_duplicate_or_training_targets(validation_samples):
    with pytest.raises(ValueError):
        validate(CalibrationProfile.fit(samples()), validation_samples)


@pytest.mark.parametrize("sample_count", [0, 2, 3.0, True])
def test_validation_requires_measured_sample_counts(sample_count):
    with pytest.raises(ValueError, match="at least three"):
        validate(CalibrationProfile.fit(samples()), sample_counts=[sample_count] * 4)


@pytest.mark.parametrize("stddev", [-0.001, float("nan"), float("inf")])
def test_validation_rejects_invalid_dispersion(stddev):
    with pytest.raises(ValueError):
        validate(CalibrationProfile.fit(samples()), raw_stddevs=[NormalizedPoint(stddev, 0.001)] * 4)


def test_validation_requires_dispersion_and_counts_for_every_target():
    with pytest.raises(ValueError, match="Every validation point"):
        validate(CalibrationProfile.fit(samples()), raw_stddevs=[NormalizedPoint(0.001, 0.001)])


@pytest.mark.parametrize("errors", [[0.13] * 4, [0.21, 0, 0, 0]])
def test_mean_and_worst_point_error_are_separate_acceptance_gates(errors):
    validation_samples = [
        CalibrationSample(NormalizedPoint(x + error, y), NormalizedPoint(x, y))
        for (x, y), error in zip(VALIDATION_POINTS, errors)
    ]
    with pytest.raises(ValueError, match="Independent calibration error"):
        validate(CalibrationProfile.fit(samples()), validation_samples)


def test_validation_measures_error_before_clamping():
    profile = CalibrationProfile.fit(samples(raw_transform=lambda point: NormalizedPoint(point.x / 2, point.y)))
    validation_samples = [
        CalibrationSample(NormalizedPoint(0.65, y), NormalizedPoint(0.95, y))
        for y in (0.3, 0.4, 0.6, 0.7)
    ]
    assert all(abs(profile.apply(sample.raw).x - sample.expected.x) < 0.06 for sample in validation_samples)
    with pytest.raises(ValueError, match="Independent calibration error"):
        validate(profile, validation_samples)


@pytest.mark.parametrize("direction", [-1, 1])
def test_large_mapping_gain_rejects_raw_noise_even_when_validation_means_are_exact(direction):
    transform = lambda point: NormalizedPoint(0.5 + direction * point.x * 0.01, point.y)
    profile = CalibrationProfile.fit(samples(raw_transform=transform))
    validation_samples = samples(VALIDATION_POINTS, raw_transform=transform)
    assert abs(profile.x_scale) == pytest.approx(100)

    with pytest.raises(ValueError, match="Mapped calibration dispersion"):
        validate(profile, validation_samples)

    validated = validate(profile, validation_samples, raw_stddevs=[NormalizedPoint(0.0001, 0.001)] * 4)
    assert validated.quality.max_mapped_stddev == pytest.approx(0.01)
    assert validated.is_usable()


def test_saved_quality_cannot_claim_an_error_inconsistent_with_its_measurements():
    profile = validate(CalibrationProfile.fit(samples()))
    altered = replace(profile, quality=replace(profile.quality, mean_error=0.01))

    assert not altered.is_usable()


def test_looser_recorded_thresholds_do_not_override_current_live_acceptance():
    validation_samples = samples(VALIDATION_POINTS, raw_transform=lambda point: NormalizedPoint(point.x + 0.13, point.y))
    profile = validate(CalibrationProfile.fit(samples()), validation_samples, max_mean_error=0.15)

    assert not profile.is_usable()


def test_default_scripted_profile_remains_usable_for_mapping_but_not_live_control():
    profile = CalibrationProfile()

    assert profile.apply(NormalizedPoint(0.3, 0.6)) == NormalizedPoint(0.3, 0.6)
    assert not profile.is_usable()
    with pytest.raises(ValueError, match="Fit the calibration"):
        validate(profile)
