import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import pytest

from interaction.audio import MacOSLiveSpeechProvider, SpeechCaptureError
from interaction.persistence import RuntimePaths


class FakeRunner:
    def __init__(self):
        self.build_calls = []
        self.capture_calls = []
        self.recordings = []
        self.build_error = None
        self.capture_error = None
        self.build_returncode = 0
        self.capture_returncode = 0
        self.artifact = "valid"
        self.payload_updates = {}
        self.stdout = None

    def __call__(self, command, *, timeout):
        if command[0] == "clang":
            self.build_calls.append((command, timeout))
            output = Path(command[command.index("-o") + 1])
            if self.artifact != "missing":
                output.write_bytes(b"" if self.artifact == "empty" else b"compiled executable")
                output.chmod(0o600 if self.artifact == "nonexecutable" else 0o755)
            if self.build_error:
                raise self.build_error
            return subprocess.CompletedProcess(command, self.build_returncode, stdout="", stderr="compiler diagnostics")
        self.capture_calls.append((command, timeout))
        recording = Path(command[command.index("--recording-path") + 1])
        recording.write_bytes(b"temporary audio")
        self.recordings.append(recording)
        if self.capture_error:
            raise self.capture_error
        payload = {
            "status": "success", "transcript": "open Safari", "provider": "macos_speech",
            "locale": command[command.index("--locale") + 1],
            "duration_s": float(command[command.index("--duration") + 1]),
            "confidence": None, "used_on_device": None, "permission_state": None,
            **self.payload_updates,
        }
        stdout = json.dumps(payload) if self.stdout is None else self.stdout
        return subprocess.CompletedProcess(command, self.capture_returncode, stdout=stdout, stderr="")


@pytest.fixture
def configured(tmp_path):
    source = tmp_path / "bridge.m"
    source.write_text("int main(void) { return 0; }\n")
    info = tmp_path / "helper, usage info.plist"
    info.write_text("<?xml version='1.0'?><plist version='1.0'><dict/></plist>")
    runner = FakeRunner()
    provider = MacOSLiveSpeechProvider(
        RuntimePaths(tmp_path / "runtime"), helper_source=source, helper_info=info,
        command_runner=runner, platform="darwin", architecture="arm64", os_version="14.0",
    )
    return provider, runner


def assert_no_temporary_builds(provider):
    assert not list((provider.paths.bin_dir / "speech").glob(".speech-*"))


@pytest.mark.parametrize("platform", ["linux", "win32", "unknown"])
def test_non_macos_capture_fails_before_building_or_running(tmp_path, platform):
    runner = FakeRunner()
    provider = MacOSLiveSpeechProvider(RuntimePaths(tmp_path / "runtime"), platform=platform, command_runner=runner)

    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()

    assert error.value.code == "unsupported_platform"
    assert runner.build_calls == runner.capture_calls == []
    assert not provider.paths.base_dir.exists()


@pytest.mark.parametrize("duration", [0, -1, 30.1, float("nan"), float("inf"), True, "4", None])
def test_invalid_duration_cannot_start_compilation_or_capture(configured, duration):
    provider, runner = configured
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn(duration_s=duration)
    assert error.value.code == "invalid_capture_arguments"
    assert runner.build_calls == runner.capture_calls == []


@pytest.mark.parametrize("locale", ["", " \n ", None, 12, "en-US\0"])
def test_invalid_locale_cannot_start_compilation_or_capture(configured, locale):
    provider, runner = configured
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn(locale=locale)
    assert error.value.code == "invalid_capture_arguments"
    assert runner.build_calls == runner.capture_calls == []


@pytest.mark.parametrize("resource, code", [("helper_source", "helper_source_missing"), ("helper_info", "helper_info_missing")])
def test_missing_packaged_resource_returns_structured_error(configured, resource, code):
    provider, runner = configured
    getattr(provider, resource).unlink()
    with pytest.raises(SpeechCaptureError) as error:
        provider.ensure_helper()
    assert error.value.code == code
    assert runner.build_calls == []


def test_info_resource_defaults_to_package_even_with_custom_source(configured):
    original, runner = configured
    provider = MacOSLiveSpeechProvider(original.paths, helper_source=original.helper_source, command_runner=runner, platform="darwin")

    assert provider.helper_info.name == "macos_speech_info.plist"
    assert provider.helper_info.is_file()
    assert provider.helper_info.parent != provider.helper_source.parent


def test_compile_only_cache_includes_flags_resource_hashes_and_platform_identity(configured):
    provider, runner = configured
    binary = provider.ensure_helper()
    manifest = json.loads((binary.parent / "build.json").read_text())

    assert binary.is_file() and os.access(binary, os.X_OK)
    assert manifest["source_sha256"] == hashlib.sha256(provider.helper_source.read_bytes()).hexdigest()
    assert manifest["info_sha256"] == hashlib.sha256(provider.helper_info.read_bytes()).hexdigest()
    assert manifest["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert manifest["platform"] == "darwin"
    assert manifest["architecture"] == "arm64"
    assert manifest["os_version"] == "14.0"
    command, timeout = runner.build_calls[0]
    assert timeout == 60
    assert command[command.index("-sectcreate") - 1:command.index(str(provider.helper_info)) + 1] == [
        "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", str(provider.helper_info),
    ]
    assert runner.capture_calls == []
    assert provider.ensure_helper() == binary
    assert len(runner.build_calls) == 1
    assert_no_temporary_builds(provider)


@pytest.mark.parametrize("change", ["source", "info", "architecture", "os_version", "flags"])
def test_cache_invalidates_on_content_or_build_identity_changes_and_preserves_previous_pair(configured, monkeypatch, change):
    provider, runner = configured
    original = provider.ensure_helper()
    original_bytes = original.read_bytes()
    original_manifest = (original.parent / "build.json").read_bytes()
    if change in {"source", "info"}:
        resource = provider.helper_source if change == "source" else provider.helper_info
        old_stat = resource.stat()
        resource.write_text(resource.read_text() + "\n/* changed contents */\n")
        os.utime(resource, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    elif change == "architecture":
        provider.architecture = "x86_64"
    elif change == "os_version":
        provider.os_version = "15.0"
    else:
        flags = provider._compiler_flags()
        monkeypatch.setattr(provider, "_compiler_flags", lambda: ["-DTEST_BUILD=1", *flags])

    rebuilt = provider.ensure_helper()

    assert rebuilt != original
    assert len(runner.build_calls) == 2
    assert original.read_bytes() == original_bytes
    assert (original.parent / "build.json").read_bytes() == original_manifest
    assert_no_temporary_builds(provider)


def test_mtime_alone_does_not_trigger_rebuild(configured):
    provider, runner = configured
    binary = provider.ensure_helper()
    modified = provider.helper_source.stat().st_mtime + 100
    os.utime(provider.helper_source, (modified, modified))

    assert provider.ensure_helper() == binary
    assert len(runner.build_calls) == 1


@pytest.mark.parametrize("artifact", ["missing", "empty", "nonexecutable"])
def test_compiler_success_requires_a_real_nonempty_executable(configured, artifact):
    provider, runner = configured
    runner.artifact = artifact
    with pytest.raises(SpeechCaptureError) as error:
        provider.ensure_helper()
    assert error.value.code == "helper_build_failed"
    assert list((provider.paths.bin_dir / "speech").iterdir()) == []


@pytest.mark.parametrize("failure, code", [
    (FileNotFoundError("clang"), "helper_compiler_unavailable"),
    (PermissionError("compiler denied"), "helper_build_failed"),
    (subprocess.TimeoutExpired(["clang"], 60), "helper_build_timeout"),
])
def test_build_process_failures_are_bounded_and_structured(configured, failure, code):
    provider, runner = configured
    runner.build_error = failure
    with pytest.raises(SpeechCaptureError) as error:
        provider.ensure_helper()
    assert error.value.code == code
    assert runner.build_calls[0][1] == 60
    assert list((provider.paths.bin_dir / "speech").iterdir()) == []


def test_failed_rebuild_preserves_previous_binary_and_metadata(configured):
    provider, runner = configured
    original = provider.ensure_helper()
    original_bytes = original.read_bytes()
    metadata = (original.parent / "build.json").read_bytes()
    provider.helper_source.write_text("changed source")
    runner.build_returncode = 1

    with pytest.raises(SpeechCaptureError) as error:
        provider.ensure_helper()

    assert error.value.code == "helper_build_failed"
    assert original.read_bytes() == original_bytes
    assert (original.parent / "build.json").read_bytes() == metadata
    assert_no_temporary_builds(provider)


def test_failed_cache_repair_restores_binary_and_manifest_together(configured, monkeypatch):
    provider, _ = configured
    binary = provider.ensure_helper()
    metadata_path = binary.parent / "build.json"
    metadata_path.write_text("incomplete metadata")
    before_binary, before_metadata = binary.read_bytes(), metadata_path.read_bytes()
    replace = os.replace

    def fail_publication(source, destination):
        if Path(source).name.startswith(".speech-build-"):
            raise OSError("Simulated publish interruption")
        return replace(source, destination)

    monkeypatch.setattr("interaction.audio.macos_speech.os.replace", fail_publication)
    with pytest.raises(SpeechCaptureError) as error:
        provider.ensure_helper()

    assert error.value.code == "helper_build_failed"
    assert binary.read_bytes() == before_binary
    assert metadata_path.read_bytes() == before_metadata
    assert_no_temporary_builds(provider)


def test_modified_cached_binary_is_rebuilt_before_use(configured):
    provider, runner = configured
    binary = provider.ensure_helper()
    binary.write_bytes(b"changed executable")

    assert provider.ensure_helper() == binary
    assert binary.read_bytes() == b"compiled executable"
    assert len(runner.build_calls) == 2
    assert_no_temporary_builds(provider)


def test_concurrent_verified_cache_winner_is_reused(configured, monkeypatch):
    provider, runner = configured
    replace = os.replace

    def peer_publishes_before_replace(source, destination):
        if Path(source).name.startswith(".speech-build-"):
            shutil.copytree(source, destination)
        return replace(source, destination)

    monkeypatch.setattr("interaction.audio.macos_speech.os.replace", peer_publishes_before_replace)

    binary = provider.ensure_helper()

    assert binary.read_bytes() == b"compiled executable"
    assert len(runner.build_calls) == 1
    assert_no_temporary_builds(provider)


@pytest.mark.parametrize("duration", [0.001, 4.0, 30.0])
def test_capture_budget_includes_both_permissions_recording_recognition_and_margin(configured, duration):
    provider, runner = configured
    capture = provider.capture_turn(duration_s=duration)

    assert capture.duration_s == duration
    assert runner.capture_calls[0][1] == 20 + duration + max(15, duration + 8) + 5
    assert not runner.recordings[0].parent.exists()


@pytest.mark.parametrize("failure, expected", [
    (subprocess.TimeoutExpired(["helper"], 44), "capture_timeout"),
    (FileNotFoundError("helper"), "capture_failed"),
    (PermissionError("helper denied"), "capture_failed"),
    (KeyboardInterrupt(), None),
])
def test_parent_cleans_partial_recording_after_timeout_failure_or_interrupt(configured, failure, expected):
    provider, runner = configured
    runner.capture_error = failure
    exception = SpeechCaptureError if expected is not None else KeyboardInterrupt
    with pytest.raises(exception) as error:
        provider.capture_turn()
    if expected is not None:
        assert error.value.code == expected
    assert not runner.recordings[0].exists()
    assert not runner.recordings[0].parent.exists()


@pytest.mark.parametrize("stage", ["creation", "cleanup"])
def test_recording_storage_failures_are_structured(configured, monkeypatch, stage):
    provider, runner = configured
    provider.ensure_helper()

    class FailingRecordingDirectory:
        def __init__(self, **kwargs):
            self.directory = None

        def __enter__(self):
            if stage == "creation":
                raise OSError("Temporary storage unavailable")
            self.directory = TemporaryDirectory(prefix="speech-storage-test-")
            return self.directory.__enter__()

        def __exit__(self, *args):
            self.directory.__exit__(*args)
            raise OSError("Temporary cleanup failed")

    monkeypatch.setattr("interaction.audio.macos_speech.TemporaryDirectory", FailingRecordingDirectory)
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()

    assert error.value.code == "capture_storage_failed"
    if stage == "creation":
        assert runner.capture_calls == []
    else:
        assert not runner.recordings[0].parent.exists()


def test_literal_transcript_spacing_case_and_punctuation_are_preserved(configured):
    provider, runner = configured
    literal = "  type Hello,\n  World!  "
    runner.payload_updates = {"transcript": literal, "confidence": None}

    capture = provider.capture_turn()

    assert capture.transcript == literal
    assert capture.confidence is None
    assert capture.used_on_device is None
    assert capture.permission_state is None


@pytest.mark.parametrize("stdout", ["not JSON", "[]", "null", '"success"', '{"status":"success"}', '{"status":true}'])
def test_nonobject_or_incomplete_native_payload_is_rejected(configured, stdout):
    provider, runner = configured
    runner.stdout = stdout
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()
    assert error.value.code == "invalid_helper_output"
    assert not runner.recordings[0].parent.exists()


@pytest.mark.parametrize("field, value", [
    ("transcript", 12), ("confidence", True), ("confidence", "0.9"),
    ("confidence", float("nan")), ("confidence", float("inf")), ("confidence", -0.1), ("confidence", 1.1),
    ("locale", None), ("locale", 12), ("locale", "fr-FR"),
    ("duration_s", None), ("duration_s", False), ("duration_s", "4.0"), ("duration_s", 3.0),
    ("provider", None), ("provider", "unexpected"),
    ("used_on_device", "false"), ("used_on_device", 1),
    ("permission_state", False), ("permission_state", "denied"),
])
def test_invalid_metadata_is_not_coerced_to_a_successful_capture(configured, field, value):
    provider, runner = configured
    runner.payload_updates = {field: value}
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()
    assert error.value.code == "invalid_helper_output"


def test_overflowed_json_float_cannot_leak_nonfinite_error_metadata(configured):
    provider, runner = configured
    runner.stdout = '{"status":"success","transcript":"click","confidence":1e309}'

    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()

    assert error.value.code == "invalid_helper_output"
    assert "confidence" not in error.value.payload
    assert json.loads(json.dumps(error.value.payload, allow_nan=False)) == error.value.payload


def test_zero_confidence_is_distinct_from_unknown(configured):
    provider, runner = configured
    runner.payload_updates = {"confidence": 0.0, "used_on_device": False}

    capture = provider.capture_turn()

    assert capture.confidence == 0.0
    assert capture.used_on_device is False


def test_success_payload_with_nonzero_exit_status_is_not_accepted(configured):
    provider, runner = configured
    runner.capture_returncode = 1
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()
    assert error.value.code == "capture_failed"


def test_empty_literal_transcript_has_structured_empty_result(configured):
    provider, runner = configured
    runner.payload_updates = {"transcript": " \n "}
    with pytest.raises(SpeechCaptureError) as error:
        provider.capture_turn()
    assert error.value.code == "empty_transcript"


def test_default_runner_passes_timeout_to_subprocess(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("interaction.audio.macos_speech.subprocess.run", run)
    MacOSLiveSpeechProvider._default_runner(["helper"], timeout=44)
    assert calls == [(["helper"], {"check": False, "capture_output": True, "text": True, "timeout": 44})]
