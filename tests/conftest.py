"""Keep fake CLI sessions independent of the test runner's desktop focus."""

import pytest

from interaction.contracts import EnvironmentSnapshot
from interaction.vision import CalibrationProfile, CalibrationSample, NormalizedPoint


@pytest.fixture(autouse=True)
def isolated_cli_environment(monkeypatch):
    # Probe-specific tests call platform.environment directly. CLI tests can
    # override this hook to exercise focus transitions without OS access.
    monkeypatch.setattr(
        "interaction.app.capture_environment",
        lambda: EnvironmentSnapshot(active_app="Test App", active_window_title="Test Window"),
    )


@pytest.fixture
def validated_profile():
    from interaction.runtime.webcam import default_webcam_calibration_targets, default_webcam_validation_targets

    profile = CalibrationProfile.fit([
        CalibrationSample(step.expected, step.expected) for step in default_webcam_calibration_targets()
    ])
    return profile.validate(
        [CalibrationSample(step.expected, step.expected) for step in default_webcam_validation_targets()],
        raw_stddevs=[NormalizedPoint(0.001, 0.001)] * 4,
        sample_counts=[6] * 4, camera_index=0,
    )


@pytest.fixture(autouse=True)
def fake_calibration_screen(monkeypatch):
    class Screen:
        screen_size = (1920, 1080)

        def __init__(self):
            self.targets = []
            self.opened = False
            self.closed = False

        def open(self):
            self.opened = True

        def show_target(self, x, y, label, *, phase="fitting"):
            assert self.opened and not self.closed
            self.targets.append((x, y, label, phase))

        def wait(self, ms):
            self.pump()

        def pump(self):
            assert self.opened and not self.closed

        def close(self):
            self.closed = True

    screen = Screen()
    monkeypatch.setattr("interaction.app.CalibrationTargetOverlay", lambda: screen)
    return screen
