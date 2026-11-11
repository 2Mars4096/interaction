import json
import subprocess

import pytest

from interaction.contracts import EnvironmentSnapshot, PlatformName
from interaction.platform.environment import capture_environment


def completed(payload, *, returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=json.dumps(payload), stderr="")


def test_probe_reads_foreground_context_with_bounded_read_only_command():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return completed({"active_app": "Safari", "active_window_title": 'A "quoted" title\n下一页'})

    snapshot = capture_environment(platform="darwin", runner=runner, timeout_s=0.75)
    assert snapshot == EnvironmentSnapshot(
        platform=PlatformName.MACOS, active_app="Safari", active_window_title='A "quoted" title\n下一页'
    )
    command, options = calls[0]
    assert command[:4] == ["osascript", "-l", "JavaScript", "-e"]
    assert options == {"check": False, "capture_output": True, "text": True, "timeout": 0.75}
    assert "activate(" not in command[-1]


def test_windowless_app_preserves_app_without_inventing_title():
    snapshot = capture_environment(
        platform="darwin", runner=lambda *args, **kwargs: completed({"active_app": "Finder", "active_window_title": None})
    )
    assert snapshot.active_app == "Finder"
    assert snapshot.active_window_title is None


@pytest.mark.parametrize("payload", [[], None, "Safari", {}, {"active_app": 3}, {"active_app": "  "}, {"active_window_title": "Untitled"}])
def test_invalid_payload_never_fabricates_context(payload):
    assert capture_environment(platform="darwin", runner=lambda *args, **kwargs: completed(payload)) == EnvironmentSnapshot()


def test_nonstring_window_is_unknown_while_valid_app_remains():
    snapshot = capture_environment(
        platform="darwin", runner=lambda *args, **kwargs: completed({"active_app": "Notes", "active_window_title": ["invalid"]})
    )
    assert snapshot.active_app == "Notes"
    assert snapshot.active_window_title is None


@pytest.mark.parametrize("error", [FileNotFoundError("osascript"), PermissionError("denied"), subprocess.TimeoutExpired("osascript", 1)])
def test_unavailable_probe_returns_unknown(error):
    def runner(*args, **kwargs):
        raise error

    assert capture_environment(platform="darwin", runner=runner) == EnvironmentSnapshot()


def test_permission_failure_does_not_reuse_successful_output():
    result = completed({"active_app": "Safari", "active_window_title": "old"}, returncode=1)
    assert capture_environment(platform="darwin", runner=lambda *args, **kwargs: result) == EnvironmentSnapshot()


def test_malformed_json_returns_unknown():
    result = subprocess.CompletedProcess([], 0, stdout="not JSON", stderr="")
    assert capture_environment(platform="darwin", runner=lambda *args, **kwargs: result) == EnvironmentSnapshot()


@pytest.mark.parametrize("platform, expected", [("linux", PlatformName.LINUX), ("win32", PlatformName.WINDOWS), ("other", PlatformName.UNKNOWN)])
def test_other_platforms_never_invoke_macos_probe(platform, expected):
    def runner(*args, **kwargs):
        raise AssertionError("Non-macOS context inspection must not run osascript")

    assert capture_environment(platform=platform, runner=runner) == EnvironmentSnapshot(platform=expected)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_cannot_start_unbounded_probe(timeout):
    def runner(*args, **kwargs):
        raise AssertionError("invalid timeout")

    assert capture_environment(platform="darwin", runner=runner, timeout_s=timeout) == EnvironmentSnapshot()
