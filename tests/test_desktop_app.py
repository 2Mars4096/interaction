import argparse
from pathlib import Path
import subprocess
import sys

import pytest

from interaction.app import _build_parser, _effective_dry_run
from interaction.desktop_worker import command_for, failure_messages
from interaction.persistence import UserSettings


def test_all_desktop_modes_map_to_valid_cli_commands(tmp_path):
    for mode in ("voice", "dictation", "gaze", "fusion", "calibrate"):
        for execute in (False, True):
            args = _build_parser().parse_args(command_for(mode, tmp_path, execute))
            assert args.runtime_dir == str(tmp_path)
            if mode == "calibrate":
                assert args.interactive
            else:
                assert _effective_dry_run(UserSettings(dry_run=False), args) is not execute


def test_preview_override_wins_even_with_conflicting_execute():
    assert _effective_dry_run(UserSettings(dry_run=False), argparse.Namespace(execute=True, dry_run=True))


def test_desktop_worker_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        command_for("arbitrary-shell-command", tmp_path, True)


def test_failure_reporting_includes_nested_capture_and_execution_errors():
    payload = {
        "live_capture": {"status": "error", "message": "Microphone denied"},
        "events": [{"result": {"status": "failure", "message": "Accessibility denied"}}],
    }
    assert failure_messages(payload) == ["Microphone denied", "Accessibility denied"]


def test_desktop_worker_check_starts_without_using_devices(tmp_path):
    import os
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "interaction.desktop_worker", "--mode", "check",
                             "--runtime-dir", str(tmp_path), "--delay", "0"],
                            capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "READY\t" in result.stdout
    assert "Ready. Choose a mode" in result.stdout


def test_native_stop_can_interrupt_worker_group_before_capture(tmp_path):
    import os
    import signal
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    process = subprocess.Popen([sys.executable, "-u", "-m", "interaction.desktop_worker", "--mode", "voice",
                                "--runtime-dir", str(tmp_path), "--delay", "30"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert process.stdout.readline().startswith("READY\t")
        assert process.stdout.readline().startswith("STATUS\tStarting")
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGINT)
        output, error = process.communicate(timeout=5)
        assert process.returncode == 0, error
        assert "Stopped" in output
        assert not list(tmp_path.glob("**/*.jsonl"))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_interactive_calibration_refuses_invisible_overlay(monkeypatch, tmp_path):
    import io
    import json
    from contextlib import redirect_stdout
    from interaction.app import main

    class Overlay:
        def __init__(self, **kwargs):
            pass

        def open(self):
            raise RuntimeError("Display unavailable")

    monkeypatch.setattr("interaction.app.CalibrationDotOverlay", Overlay)
    output = io.StringIO()
    with redirect_stdout(output):
        main(["gaze-calibrate", "--interactive", "--runtime-dir", str(tmp_path)])
    payload = json.loads(output.getvalue())
    assert payload["calibration"]["error"] == "calibration_overlay_unavailable"
    assert not (tmp_path / "profiles/webcam-live.json").exists()
