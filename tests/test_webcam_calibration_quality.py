import pytest

from interaction.runtime.webcam import (
    WebcamCalibrationError, collect_webcam_calibration,
    default_webcam_calibration_targets, default_webcam_validation_targets,
)
from interaction.vision import GazeSample, NormalizedPoint, WebcamSampleAggregate


class Provider:
    camera_index = 0

    def __init__(self, *, transform=lambda point: point, validation_drift=0.0, stddev=.001, used_frames=6):
        self.samples = []
        self.calls = 0
        for target in (*default_webcam_calibration_targets(), *default_webcam_validation_targets()):
            point = transform(target.expected)
            if target.phase == "validation":
                point = NormalizedPoint(point.x + validation_drift, point.y)
            self.samples.append(WebcamSampleAggregate(
                GazeSample(point, confidence=.95), used_frames, 6,
                raw_stddev=NormalizedPoint(stddev, stddev), sample_span_ms=500,
            ))

    def capture_average_sample(self, **kwargs):
        self.calls += 1
        callback = kwargs.get("on_frame")
        if callback:
            callback()
        return self.samples.pop(0)


def test_complete_samples_validate_on_different_points_without_refitting():
    provider = Provider(transform=lambda point: NormalizedPoint(1 - point.x, .1 + .8 * point.y))
    profile, steps = collect_webcam_calibration(provider)
    assert profile.x_scale == pytest.approx(-1)
    assert profile.x_offset == pytest.approx(1)
    assert profile.y_scale == pytest.approx(1.25)
    assert profile.y_offset == pytest.approx(-.125)
    assert profile.is_usable(camera_index=0)
    assert provider.calls == 9
    fitting = {(item["expected"]["x"], item["expected"]["y"]) for item in steps if item["phase"] == "fitting"}
    validation = {(item["expected"]["x"], item["expected"]["y"]) for item in steps if item["phase"] == "validation"}
    assert len(fitting) == 5 and len(validation) == 4
    assert fitting.isdisjoint(validation)


def test_partial_average_cannot_make_a_calibration_step_successful():
    provider = Provider(used_frames=1)
    with pytest.raises(WebcamCalibrationError) as failure:
        collect_webcam_calibration(provider)
    assert failure.value.code == "insufficient_calibration_samples"
    assert provider.calls == 5
    assert all(not step["success"] for step in failure.value.payload["steps"])


def test_constant_signal_cannot_create_usable_mapping():
    with pytest.raises(WebcamCalibrationError) as failure:
        collect_webcam_calibration(Provider(transform=lambda point: NormalizedPoint(.5, .5)))
    assert failure.value.code == "invalid_calibration_signal"


def test_perfect_means_do_not_hide_noise_amplified_by_mapping():
    provider = Provider(transform=lambda point: NormalizedPoint(.49 + point.x * .02, .49 + point.y * .02), stddev=.002)
    with pytest.raises(WebcamCalibrationError) as failure:
        collect_webcam_calibration(provider)
    assert failure.value.code == "unstable_calibration_samples"
    assert provider.calls == 5


def test_independent_validation_drift_rejects_good_training_fit():
    with pytest.raises(WebcamCalibrationError) as failure:
        collect_webcam_calibration(Provider(validation_drift=.15))
    assert failure.value.code == "calibration_validation_failed"
    assert len(failure.value.payload["steps"]) == 9


@pytest.mark.parametrize("changes", [{"frames_per_step": 1}, {"delta_ms": 0}, {"max_attempts_multiplier": 0}])
def test_invalid_capture_configuration_is_structured(changes):
    provider = Provider()
    with pytest.raises(WebcamCalibrationError) as failure:
        collect_webcam_calibration(provider, **changes)
    assert failure.value.code == "invalid_calibration_options"
    assert provider.calls == 0
