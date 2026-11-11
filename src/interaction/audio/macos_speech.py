"""macOS live speech capture via a small Objective-C bridge."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import hashlib
from pathlib import Path
import json
import math
import os
import platform as host_platform
import subprocess
import sys
from tempfile import TemporaryDirectory

from interaction.persistence import RuntimePaths


MAX_CAPTURE_DURATION_S = 30.0
BUILD_TIMEOUT_S = 60.0
PERMISSION_BUDGET_S = 20.0
PROCESS_MARGIN_S = 5.0


@dataclass(frozen=True)
class LiveSpeechCapture:
    transcript: str
    confidence: float | None
    locale: str
    duration_s: float
    provider: str = "macos_speech"
    used_on_device: bool | None = None
    permission_state: str | None = None
    capture_mode: str = "recorded"
    endpoint_reason: str | None = None
    max_duration_s: float | None = None


@dataclass(frozen=True)
class SpeechStreamEvent:
    """A display-only native speech update; executable text is returned separately."""

    kind: str
    transcript: str | None = None
    elapsed_s: float | None = None


class SpeechCaptureError(RuntimeError):
    """Raised when the native speech helper cannot produce a transcript."""

    def __init__(self, code: str, message: str, *, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


class MacOSLiveSpeechProvider:
    """Compile and invoke the macOS speech bridge on demand."""

    def __init__(
        self,
        paths: RuntimePaths,
        *,
        helper_source: Path | None = None,
        helper_info: Path | None = None,
        command_runner=None,
        process_factory=None,
        platform: str | None = None,
        architecture: str | None = None,
        os_version: str | None = None,
    ) -> None:
        self.paths = paths
        self.helper_source = Path(helper_source) if helper_source is not None else Path(__file__).with_name("macos_speech_bridge.m")
        self.helper_info = Path(helper_info) if helper_info is not None else Path(__file__).with_name("macos_speech_info.plist")
        self.command_runner = command_runner or self._default_runner
        self.process_factory = process_factory or subprocess.Popen
        self.platform = sys.platform if platform is None else platform
        self.architecture = host_platform.machine() if architecture is None else architecture
        self.os_version = (host_platform.mac_ver()[0] or host_platform.release()) if os_version is None else os_version

    @property
    def helper_binary_path(self) -> Path:
        return self._cache_directory(self._build_identity()) / "macos_speech_bridge"

    def ensure_helper(self) -> Path:
        """Compile only; publish a complete source/platform-specific cache entry."""
        identity = self._build_identity()
        cache_directory = self._cache_directory(identity)
        binary = cache_directory / "macos_speech_bridge"
        try:
            self.paths.ensure()
            cache_directory.parent.mkdir(parents=True, exist_ok=True)
            if self._cached_helper_is_valid(cache_directory, identity):
                return binary
            with TemporaryDirectory(prefix=".speech-build-", dir=cache_directory.parent) as staging:
                staging_directory = Path(staging)
                candidate = staging_directory / "macos_speech_bridge"
                self._compile_helper(candidate)
                if not self._binary_is_valid(candidate):
                    raise SpeechCaptureError("helper_build_failed", "The compiler did not produce a nonempty executable speech helper.")
                if self._build_identity() != identity:
                    raise SpeechCaptureError("helper_source_changed", "The speech helper resources changed while compiling. Please retry.")
                with candidate.open("rb+") as built_file:
                    os.fsync(built_file.fileno())
                manifest = {**identity, "binary_sha256": _sha256(candidate.read_bytes())}
                with (staging_directory / "build.json").open("w", encoding="utf-8") as metadata:
                    json.dump(manifest, metadata, sort_keys=True, indent=2, allow_nan=False)
                    metadata.write("\n")
                    metadata.flush()
                    os.fsync(metadata.fileno())
                self._publish_cache(staging_directory, cache_directory, identity)
        except SpeechCaptureError:
            raise
        except OSError as error:
            raise SpeechCaptureError("helper_build_failed", "Unable to prepare or publish the macOS speech helper.", payload={"detail": str(error)}) from error
        return binary

    def _compile_helper(self, output: Path) -> None:
        command = [
            "clang",
            *self._compiler_flags(),
            str(self.helper_info),
            str(self.helper_source),
            "-o",
            str(output),
        ]
        try:
            completed = self.command_runner(command, timeout=BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise SpeechCaptureError("helper_build_timeout", "Compiling the macOS speech helper exceeded its time limit.") from error
        except FileNotFoundError as error:
            raise SpeechCaptureError("helper_compiler_unavailable", "The clang compiler is unavailable. Install the macOS Command Line Tools.") from error
        except OSError as error:
            raise SpeechCaptureError("helper_build_failed", "Unable to run the macOS speech helper compiler.", payload={"detail": str(error)}) from error
        if completed.returncode != 0:
            raise SpeechCaptureError(
                "helper_build_failed",
                "Failed to build the macOS speech helper.",
                payload={"stdout": _output_text(completed.stdout), "stderr": _output_text(completed.stderr)},
            )

    def _compiler_flags(self) -> list[str]:
        return [
            "-fobjc-arc", "-arch", self.architecture,
            "-framework", "Foundation", "-framework", "Speech", "-framework", "AVFoundation",
            "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker",
        ]

    def _build_identity(self) -> dict[str, object]:
        if self.platform != "darwin":
            raise SpeechCaptureError("unsupported_platform", "Native live speech capture is available on macOS only.", payload={"platform": self.platform})
        resources = []
        for path, code in ((self.helper_source, "helper_source_missing"), (self.helper_info, "helper_info_missing")):
            try:
                contents = path.read_bytes()
            except OSError as error:
                raise SpeechCaptureError(code, f"A packaged speech helper resource is unavailable: {path.name}.", payload={"path": str(path)}) from error
            if not contents:
                raise SpeechCaptureError(code, f"A packaged speech helper resource is empty: {path.name}.")
            resources.append(_sha256(contents))
        return {
            "schema_version": 1,
            "source_sha256": resources[0], "info_sha256": resources[1],
            "platform": self.platform, "architecture": self.architecture, "os_version": self.os_version,
            "compiler_flags": [*self._compiler_flags(), "<embedded-info-plist>"],
        }

    def _cache_directory(self, identity: dict[str, object]) -> Path:
        key = _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return self.paths.bin_dir / "speech" / key

    @staticmethod
    def _binary_is_valid(binary: Path) -> bool:
        try:
            return binary.is_file() and not binary.is_symlink() and binary.stat().st_size > 0 and os.access(binary, os.X_OK)
        except OSError:
            return False

    def _cached_helper_is_valid(self, directory: Path, identity: dict[str, object]) -> bool:
        binary = directory / "macos_speech_bridge"
        if not self._binary_is_valid(binary):
            return False
        try:
            manifest = json.loads((directory / "build.json").read_text(encoding="utf-8"))
            return manifest == {**identity, "binary_sha256": _sha256(binary.read_bytes())}
        except (OSError, UnicodeError, ValueError, TypeError):
            return False

    def _publish_cache(self, staging: Path, directory: Path, identity: dict[str, object]) -> None:
        if directory.exists():
            # A concurrent builder may have finished the same identity already.
            if self._cached_helper_is_valid(directory, identity):
                return
            # Repair an incomplete entry only after a complete replacement is
            # ready. Restore both its binary and metadata if publication fails.
            with TemporaryDirectory(prefix=".speech-previous-", dir=directory.parent) as previous:
                backup = Path(previous) / "cache"
                os.replace(directory, backup)
                try:
                    os.replace(staging, directory)
                except BaseException:
                    os.replace(backup, directory)
                    raise
        else:
            try:
                os.replace(staging, directory)
            except OSError:
                # Another process can publish after the initial existence
                # check. Its complete verified entry is equally usable.
                if not self._cached_helper_is_valid(directory, identity):
                    raise

    def capture_turn(self, *, duration_s: float = 4.0, locale: str = "en-US") -> LiveSpeechCapture:
        if not _finite_number(duration_s) or not 0 < duration_s <= MAX_CAPTURE_DURATION_S:
            raise SpeechCaptureError("invalid_capture_arguments", "Speech capture duration must be finite, positive, and at most 30 seconds.")
        if not isinstance(locale, str) or not locale.strip() or "\0" in locale:
            raise SpeechCaptureError("invalid_capture_arguments", "Speech capture requires a nonempty locale identifier.")
        locale = locale.strip()
        duration_s = float(duration_s)
        binary = self.ensure_helper()
        timeout = PERMISSION_BUDGET_S + duration_s + max(15.0, duration_s + 8.0) + PROCESS_MARGIN_S
        # The parent owns the recording directory. Subprocess timeout or Ctrl-C
        # can bypass native cleanup, but still exits this context manager.
        try:
            with TemporaryDirectory(prefix="interaction-speech-") as recording_directory:
                recording_path = Path(recording_directory) / "recording.m4a"
                try:
                    completed = self.command_runner(
                        [str(binary), "--duration", format(duration_s, ".17g"), "--locale", locale,
                         "--recording-path", str(recording_path)],
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired as error:
                    raise SpeechCaptureError("capture_timeout", "The native speech helper exceeded its capture and recognition time limit.") from error
                except OSError as error:
                    raise SpeechCaptureError("capture_failed", "Unable to run the native speech helper.", payload={"detail": str(error)}) from error
        except OSError as error:
            raise SpeechCaptureError("capture_storage_failed", "Unable to create or clean the temporary speech recording directory.", payload={"detail": str(error)}) from error
        payload = self._parse_payload(completed.stdout, completed.stderr)
        if payload.get("status") == "error":
            if not _nonempty_string(payload.get("error")) or not _nonempty_string(payload.get("message")):
                raise _invalid_payload(payload, "The speech helper returned malformed error metadata.")
            raise SpeechCaptureError(
                payload["error"], payload["message"], payload=payload,
            )
        if payload.get("status") != "success":
            raise _invalid_payload(payload, "The speech helper did not return a recognized status.")
        if completed.returncode != 0:
            raise SpeechCaptureError("capture_failed", "The speech helper exited unsuccessfully despite a success payload.", payload=payload)
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise _invalid_payload(payload, "The speech helper transcript must be text.")
        if not transcript.strip():
            raise SpeechCaptureError(
                "empty_transcript",
                "The speech helper completed but returned an empty transcript.",
                payload=payload,
            )
        raw_confidence = payload.get("confidence")
        if raw_confidence is not None and (not _finite_number(raw_confidence) or not 0 <= raw_confidence <= 1):
            raise _invalid_payload(payload, "The speech helper confidence must be unknown or a finite value between zero and one.")
        if payload.get("locale") != locale or not _nonempty_string(payload.get("locale")):
            raise _invalid_payload(payload, "The speech helper returned an invalid or mismatched locale.")
        reported_duration = payload.get("duration_s")
        if not _finite_number(reported_duration) or not math.isclose(reported_duration, duration_s, rel_tol=0, abs_tol=1e-6):
            raise _invalid_payload(payload, "The speech helper returned an invalid or mismatched duration.")
        if payload.get("provider") != "macos_speech":
            raise _invalid_payload(payload, "The speech helper returned an invalid provider identity.")
        used_on_device = payload.get("used_on_device")
        if used_on_device is not None and type(used_on_device) is not bool:
            raise _invalid_payload(payload, "The speech helper on-device flag must be a boolean or unknown.")
        permission_state = payload.get("permission_state")
        if permission_state is not None and permission_state != "granted":
            raise _invalid_payload(payload, "A successful speech capture must report granted or unknown permission state.")
        return LiveSpeechCapture(
            transcript=transcript,
            confidence=float(raw_confidence) if raw_confidence is not None else None,
            locale=locale, duration_s=duration_s,
            provider="macos_speech", used_on_device=used_on_device, permission_state=permission_state,
        )

    def capture_stream(
        self,
        *,
        duration_s: float = 4.0,
        locale: str = "en-US",
        silence_duration_s: float = 0.8,
        on_event: Callable[[SpeechStreamEvent], None] | None = None,
    ) -> LiveSpeechCapture:
        """Recognize live buffers, publishing previews but returning only a validated final."""
        from .streaming import run_speech_stream, validate_stream_arguments

        duration, language, silence = validate_stream_arguments(duration_s, locale, silence_duration_s)
        binary = self.ensure_helper()
        return run_speech_stream(
            binary, duration_s=duration, locale=language, silence_duration_s=silence,
            process_factory=self.process_factory, on_event=on_event,
        )

    @staticmethod
    def _default_runner(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)

    @staticmethod
    def _parse_payload(stdout: str, stderr: str) -> dict[str, object]:
        for candidate in (_output_text(stdout), _output_text(stderr)):
            if not candidate:
                continue
            try:
                payload = json.loads(candidate, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
                if isinstance(payload, dict):
                    return payload
            except (ValueError, TypeError):
                continue
        return {
            "status": "error",
            "error": "invalid_helper_output",
            "message": "The speech helper did not return valid JSON output.",
            "stdout": _output_text(stdout),
            "stderr": _output_text(stderr),
        }


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _finite_number(value: object) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _output_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value {value} is invalid helper output.")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON number is invalid helper output.")
    return parsed


def _invalid_payload(payload: dict[str, object], message: str) -> SpeechCaptureError:
    return SpeechCaptureError("invalid_helper_output", message, payload=payload)
