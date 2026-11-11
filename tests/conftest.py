"""Keep fake CLI sessions independent of the test runner's desktop focus."""

import pytest

from interaction.contracts import EnvironmentSnapshot


@pytest.fixture(autouse=True)
def isolated_cli_environment(monkeypatch):
    # Probe-specific tests call platform.environment directly. CLI tests can
    # override this hook to exercise focus transitions without OS access.
    monkeypatch.setattr(
        "interaction.app.capture_environment",
        lambda: EnvironmentSnapshot(active_app="Test App", active_window_title="Test Window"),
    )
