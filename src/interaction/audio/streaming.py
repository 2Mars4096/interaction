"""Bounded, final-only transport for the native speech JSONL protocol."""

from __future__ import annotations

from collections.abc import Callable
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time

from .macos_speech import LiveSpeechCapture, SpeechCaptureError, SpeechStreamEvent


PERMISSION_BUDGET_S = 20.0
STREAM_FINALIZATION_TIMEOUT_S = 8.0
PROCESS_MARGIN_S = 5.0
STREAM_TERMINATE_TIMEOUT_S = 1.0
CAPTURE_SCHEDULING_TOLERANCE_S = 0.5
MAX_STREAM_LINE_BYTES = 64 * 1024
MAX_STREAM_STDOUT_BYTES = 1024 * 1024
MAX_STREAM_EVENTS = 1024
MAX_STREAM_STDERR_BYTES = 16 * 1024


def validate_stream_arguments(duration_s, locale, silence_duration_s) -> tuple[float, str, float]:
    if not _finite_number(duration_s) or not 0 < duration_s <= 30:
        raise SpeechCaptureError("invalid_capture_arguments", "Speech capture duration must be finite, positive, and at most 30 seconds.")
    if not isinstance(locale, str) or not locale.strip() or "\0" in locale or len(locale) > 128:
        raise SpeechCaptureError("invalid_capture_arguments", "Speech capture requires a nonempty locale identifier of at most 128 characters.")
    if not _finite_number(silence_duration_s) or not 0.2 <= silence_duration_s <= 5:
        raise SpeechCaptureError("invalid_capture_arguments", "Speech silence duration must be finite and between 0.2 and 5 seconds.")
    return float(duration_s), locale.strip(), float(silence_duration_s)


class SpeechStreamDecoder:
    """Validate ordering and metadata without making partial text executable."""

    def __init__(self, *, duration_s: float, locale: str, silence_duration_s: float) -> None:
        self.duration_s = duration_s
        self.locale = locale
        self.silence_duration_s = silence_duration_s
        self.sequence = 0
        self.ready = False
        self.used_on_device: bool | None = None
        self.last_elapsed_s = 0.0
        self.result: LiveSpeechCapture | None = None

    def accept(self, line: bytes) -> SpeechStreamEvent | None:
        if self.result is not None:
            raise _invalid("The speech helper emitted output after its final result.")
        if not line or len(line) > MAX_STREAM_LINE_BYTES:
            raise _invalid("The speech helper emitted an empty or oversized event.")
        try:
            payload = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                                 parse_constant=_reject_constant, parse_float=_finite_json_float)
        except (UnicodeError, ValueError, TypeError, RecursionError) as error:
            raise _invalid("The speech helper emitted malformed JSONL.") from error
        if not isinstance(payload, dict):
            raise _invalid("Each speech event must be a JSON object.")
        if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
            raise _invalid("The speech helper returned an unsupported stream version.")
        sequence = payload.get("sequence")
        if type(sequence) is not int or sequence != self.sequence + 1 or sequence > MAX_STREAM_EVENTS:
            raise _invalid("Speech events are out of sequence or exceed the event limit.")
        if payload.get("provider") != "macos_speech" or payload.get("locale") != self.locale:
            raise _invalid("The speech helper returned mismatched provider or locale metadata.")
        self.sequence = sequence
        event = payload.get("event")
        if event == "error":
            if not _text(payload.get("error")) or not _text(payload.get("message")):
                raise _invalid("The speech helper returned malformed error metadata.")
            raise SpeechCaptureError(payload["error"], payload["message"], payload=payload)
        if event == "ready":
            if self.ready:
                raise _invalid("The speech helper announced readiness more than once.")
            if not _matches(payload.get("max_duration_s"), self.duration_s) or not _matches(payload.get("silence_duration_s"), self.silence_duration_s):
                raise _invalid("The speech helper returned mismatched capture settings.")
            self._check_permission(payload)
            self.used_on_device = payload["used_on_device"]
            self.ready = True
            return SpeechStreamEvent(kind="ready")
        if not self.ready:
            raise _invalid("The speech helper emitted recognition before audio readiness.")
        if event == "partial":
            text = payload.get("transcript")
            elapsed = payload.get("elapsed_s")
            if not _text(text):
                raise _invalid("A partial speech transcript must be nonempty text.")
            upper = self.duration_s + CAPTURE_SCHEDULING_TOLERANCE_S + STREAM_FINALIZATION_TIMEOUT_S
            if not _finite_number(elapsed) or not self.last_elapsed_s <= elapsed <= upper:
                raise _invalid("The speech helper returned an invalid or out-of-order partial timestamp.")
            self.last_elapsed_s = float(elapsed)
            return SpeechStreamEvent(kind="partial", transcript=text, elapsed_s=float(elapsed))
        if event != "final":
            raise _invalid("The speech helper returned an unrecognized stream event.")
        if not _text(payload.get("transcript")):
            raise _invalid("A final speech transcript must be nonempty text.")
        if "confidence" not in payload:
            raise _invalid("The final speech event must report confidence or explicit null.")
        confidence = payload["confidence"]
        if confidence is not None and (not _finite_number(confidence) or not 0 <= confidence <= 1):
            raise _invalid("Speech confidence must be unknown or finite between zero and one.")
        duration = payload.get("duration_s")
        if not _finite_number(duration) or not 0 < duration <= self.duration_s + CAPTURE_SCHEDULING_TOLERANCE_S:
            raise _invalid("The speech helper returned an invalid measured capture duration.")
        if not _matches(payload.get("max_duration_s"), self.duration_s):
            raise _invalid("The final speech event returned a mismatched capture maximum.")
        endpoint_reason = payload.get("endpoint_reason")
        if not isinstance(endpoint_reason, str) or endpoint_reason not in {"silence", "recognizer_final"}:
            raise _invalid("The speech helper did not finish at a valid utterance endpoint.")
        self._check_permission(payload)
        if payload["used_on_device"] != self.used_on_device:
            raise _invalid("Speech recognition mode changed within the same turn.")
        self.result = LiveSpeechCapture(
            transcript=payload["transcript"], confidence=float(confidence) if confidence is not None else None,
            locale=self.locale, duration_s=float(duration), provider="macos_speech",
            used_on_device=self.used_on_device, permission_state="granted", capture_mode="stream",
            endpoint_reason=payload["endpoint_reason"], max_duration_s=self.duration_s,
        )
        return None

    @staticmethod
    def _check_permission(payload: dict) -> None:
        if payload.get("permission_state") != "granted" or type(payload.get("used_on_device")) is not bool:
            raise _invalid("The speech helper returned invalid permission or recognition mode metadata.")


def run_speech_stream(
    binary: Path,
    *,
    duration_s: float,
    locale: str,
    silence_duration_s: float,
    process_factory=subprocess.Popen,
    on_event: Callable[[SpeechStreamEvent], None] | None = None,
) -> LiveSpeechCapture:
    """Drain both pipes with fixed caps; require a valid final, EOF, and exit zero."""
    command = [str(binary), "--stream", "--duration", format(duration_s, ".17g"),
               "--locale", locale, "--silence-duration", format(silence_duration_s, ".17g")]
    try:
        process = process_factory(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
    except OSError as error:
        raise SpeechCaptureError("capture_failed", "Unable to start native streaming speech.", payload={"detail": str(error)}) from error
    decoder = SpeechStreamDecoder(duration_s=duration_s, locale=locale, silence_duration_s=silence_duration_s)
    pending = bytearray()
    stderr_tail = bytearray()
    stdout_bytes = 0
    deadline = time.monotonic() + PERMISSION_BUDGET_S + duration_s + STREAM_FINALIZATION_TIMEOUT_S + PROCESS_MARGIN_S
    try:
        if process.stdout is None or process.stderr is None:
            raise SpeechCaptureError("capture_failed", "The native speech helper did not provide protocol pipes.")
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SpeechCaptureError("capture_timeout", "Streaming speech exceeded its capture and finalization time limit.",
                                             payload={"stderr": stderr_tail.decode("utf-8", errors="replace")})
                for key, _mask in selector.select(min(0.1, remaining)):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        stderr_tail.extend(chunk)
                        if len(stderr_tail) > MAX_STREAM_STDERR_BYTES:
                            del stderr_tail[:-MAX_STREAM_STDERR_BYTES]
                        continue
                    stdout_bytes += len(chunk)
                    if stdout_bytes > MAX_STREAM_STDOUT_BYTES:
                        raise _invalid("The speech helper exceeded the total output limit.")
                    pending.extend(chunk)
                    while (newline := pending.find(b"\n")) >= 0:
                        if newline > MAX_STREAM_LINE_BYTES:
                            raise _invalid("The speech helper exceeded the event line limit.")
                        line = bytes(pending[:newline])
                        del pending[:newline + 1]
                        event = decoder.accept(line)
                        if event is not None and on_event is not None:
                            on_event(event)
                    if len(pending) > MAX_STREAM_LINE_BYTES:
                        raise _invalid("The speech helper exceeded the unfinished event line limit.")
        if pending:
            raise _invalid("The speech helper ended with an incomplete JSONL record.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SpeechCaptureError("capture_timeout", "Streaming speech did not finish before its deadline.")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise SpeechCaptureError("capture_timeout", "The speech helper did not exit after closing its output.") from error
        if returncode != 0:
            raise SpeechCaptureError("capture_failed", "The speech helper exited unsuccessfully; no command was accepted.",
                                     payload={"returncode": returncode, "stderr": stderr_tail.decode("utf-8", errors="replace")})
        if decoder.result is None:
            raise _invalid("The speech helper ended without a final transcript.")
        return decoder.result
    except OSError as error:
        raise SpeechCaptureError("capture_failed", "Unable to read native speech output.", payload={"detail": str(error)}) from error
    finally:
        _close_process(process)


def _close_process(process) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=STREAM_TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=STREAM_TERMINATE_TIMEOUT_S)
                except subprocess.TimeoutExpired as error:
                    raise SpeechCaptureError("capture_shutdown_timeout", "The speech helper did not stop after cancellation.") from error
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def _invalid(message: str) -> SpeechCaptureError:
    return SpeechCaptureError("invalid_helper_output", message)


def _finite_number(value: object) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _matches(value: object, expected: float) -> bool:
    return _finite_number(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-6)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated JSON keys are not permitted in speech events.")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value {value} is invalid.")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON number is invalid.")
    return parsed
