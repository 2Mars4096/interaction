"""Validate real child pipes and speech protocol failures without microphone access."""

import json
import subprocess
import sys
import time

import pytest

from interaction.audio import MacOSLiveSpeechProvider, SpeechCaptureError
from interaction.persistence import RuntimePaths


def envelope(kind, sequence, **fields):
    return {"schema_version": 1, "sequence": sequence, "provider": "macos_speech",
            "locale": "en-US", "event": kind, **fields}


def ready(*, duration=4.0, **changes):
    return {**envelope("ready", 1, max_duration_s=duration, silence_duration_s=0.8,
                       permission_state="granted", used_on_device=True), **changes}


def partial(text="open Mail", sequence=2, elapsed=0.4, **changes):
    return {**envelope("partial", sequence, transcript=text, elapsed_s=elapsed), **changes}


def final(*, sequence=2, duration=1.2, maximum=4.0, **changes):
    return {**envelope("final", sequence, transcript="open Safari", confidence=None,
                       duration_s=duration, max_duration_s=maximum, endpoint_reason="silence",
                       permission_state="granted", used_on_device=True), **changes}


def emit_script(events):
    serialized = json.dumps(events, ensure_ascii=False)
    return f"for event in json.loads({serialized!r}):\n    print(json.dumps(event, ensure_ascii=False), flush=True)\n"


@pytest.fixture
def speech_child(tmp_path, monkeypatch):
    children = []

    def create(body):
        script = tmp_path / f"speech_child_{len(children)}.py"
        script.write_text("import json, os, pathlib, signal, sys, time\n" + body, encoding="utf-8")
        commands = []
        processes = []

        def factory(command, **kwargs):
            commands.append(command)
            process = subprocess.Popen([sys.executable, "-I", "-u", str(script)], **kwargs)
            processes.append(process)
            children.append(process)
            return process

        provider = MacOSLiveSpeechProvider(RuntimePaths(tmp_path / "runtime"),
                                           platform="darwin", process_factory=factory)
        monkeypatch.setattr(provider, "ensure_helper", lambda: tmp_path / "unused_native_helper")
        return provider, processes, commands

    yield create
    # Assertion failures must not leave the harmless child scripts running.
    for process in children:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def assert_reaped(processes):
    assert len(processes) == 1
    process = processes[0]
    assert process.poll() is not None
    assert process.stdout.closed and process.stderr.closed


def shorten_deadline(monkeypatch):
    from interaction.audio import streaming

    for name in ("PERMISSION_BUDGET_S", "STREAM_FINALIZATION_TIMEOUT_S", "PROCESS_MARGIN_S"):
        monkeypatch.setattr(streaming, name, 0.05)
    monkeypatch.setattr(streaming, "STREAM_TERMINATE_TIMEOUT_S", 0.1)


def test_partials_arrive_before_final_and_revisions_preserve_final_text(speech_child, tmp_path):
    acknowledgement = tmp_path / "partial_seen"
    text = "type Hello,  world!\n下一行。"
    body = emit_script([ready(), partial(), partial("open Safari", sequence=3, elapsed=0.7)])
    body += f"ack = pathlib.Path({str(acknowledgement)!r})\n"
    body += "deadline = time.monotonic() + 3\nwhile not ack.exists() and time.monotonic() < deadline:\n    time.sleep(0.005)\n"
    body += "assert ack.exists(), 'The parent did not deliver the live partial callback.'\n"
    body += emit_script([final(sequence=4, transcript=text)])
    provider, processes, commands = speech_child(body)
    observed = []

    def on_event(event):
        observed.append(event)
        if event.kind == "partial" and event.transcript == "open Safari":
            assert processes[0].poll() is None
            acknowledgement.touch()

    capture = provider.capture_stream(on_event=on_event)
    assert [(event.kind, event.transcript) for event in observed] == [
        ("ready", None), ("partial", "open Mail"), ("partial", "open Safari"),
    ]
    assert capture.transcript == text
    assert capture.confidence is None
    assert capture.capture_mode == "stream"
    assert capture.duration_s == 1.2 < capture.max_duration_s == 4.0
    assert capture.endpoint_reason == "silence"
    assert capture.used_on_device is True
    assert "--stream" in commands[0]
    assert "--recording-path" not in commands[0]
    assert_reaped(processes)


def test_partial_during_finalization_can_follow_capture_endpoint(speech_child):
    provider, processes, _ = speech_child(emit_script([
        ready(), partial(elapsed=4.7), final(sequence=3, duration=2.2),
    ]))
    capture = provider.capture_stream()
    assert capture.duration_s == 2.2
    assert_reaped(processes)


@pytest.mark.parametrize("events", [
    [],
    [ready()],
    [partial()],
    [final()],
    [ready(), ready(sequence=2)],
    [ready(), partial(sequence=3), final(sequence=4)],
    [ready(), partial(sequence=True)],
    [ready(schema_version=True)],
    [ready(), final(), final(sequence=3)],
    [ready(), final(), partial(sequence=3)],
    [ready(), partial(elapsed=0.8), partial(sequence=3, elapsed=0.7), final(sequence=4)],
    [ready(), envelope("unknown", 2)],
    [ready(), partial(), envelope("error", 3, error="recognition_failed", message="No final."), final(sequence=4)],
    [None],
], ids=["empty", "missing-final", "partial-before-ready", "final-before-ready", "duplicate-ready",
         "sequence-gap", "boolean-sequence", "boolean-version", "duplicate-final", "partial-after-final",
         "regressing-elapsed", "unknown-event", "final-after-error", "non-object"])
def test_invalid_event_stream_never_returns_executable_capture(speech_child, events):
    provider, processes, _ = speech_child(emit_script(events))
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


@pytest.mark.parametrize("changes", [
    {"confidence": float("nan")}, {"confidence": float("inf")}, {"confidence": True},
    {"confidence": -0.1}, {"confidence": 1.1}, {"transcript": "   "}, {"transcript": 123},
    {"duration_s": -1}, {"duration_s": True}, {"duration_s": 4.51},
    {"max_duration_s": 5}, {"locale": "fr-FR"}, {"provider": "other"},
    {"permission_state": "denied"}, {"used_on_device": "yes"}, {"used_on_device": False},
    {"endpoint_reason": "max_duration"}, {"endpoint_reason": []}, {"endpoint_reason": {}},
], ids=["nan-confidence", "infinite-confidence", "boolean-confidence", "negative-confidence",
         "high-confidence", "empty-text", "nontext", "negative-duration", "boolean-duration",
         "excess-duration", "wrong-maximum", "wrong-locale", "wrong-provider", "denied-permission",
         "nonboolean-ondevice", "changed-ondevice", "truncated-at-ceiling", "array-endpoint", "object-endpoint"])
def test_invalid_final_metadata_never_returns_executable_capture(speech_child, changes):
    provider, processes, _ = speech_child(emit_script([ready(), final(**changes)]))
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


@pytest.mark.parametrize("body", [
    "sys.stdout.buffer.write(b'\\xff\\xfe\\n')\n",
    "print('{not json}', flush=True)\n",
    emit_script([ready()]) + "sys.stdout.write('{\"event\":\"final\"')\nsys.stdout.flush()\n",
    emit_script([ready(), final()]) + "print('trailing garbage', flush=True)\n",
])
def test_invalid_or_truncated_framing_never_returns_capture(speech_child, body):
    provider, processes, _ = speech_child(body)
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


@pytest.mark.parametrize("after_partial", [False, True])
def test_native_error_is_preserved_without_promoting_last_partial(speech_child, after_partial):
    prefix = [ready(), partial()] if after_partial else []
    code = "recognition_failed" if after_partial else "microphone_denied"
    error = envelope("error", len(prefix) + 1, error=code, message="Native capture failed.")
    provider, processes, _ = speech_child(emit_script(prefix + [error]) + "sys.exit(1)\n")
    observed = []
    with pytest.raises(SpeechCaptureError) as caught:
        provider.capture_stream(on_event=observed.append)
    assert caught.value.code == code
    assert not any(event.kind == "final" for event in observed)
    assert_reaped(processes)


def test_valid_final_does_not_override_unsuccessful_process_exit(speech_child):
    provider, processes, _ = speech_child(emit_script([ready(), final()]) + "sys.exit(7)\n")
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert processes[0].returncode == 7
    assert_reaped(processes)


def test_stderr_flood_does_not_deadlock_or_become_protocol_data(speech_child):
    body = "sys.stderr.buffer.write(b'diagnostic noise' * 100000)\nsys.stderr.flush()\n"
    provider, processes, _ = speech_child(body + emit_script([ready(), final()]))
    capture = provider.capture_stream()
    assert capture.transcript == "open Safari"
    assert_reaped(processes)


def test_failure_retains_only_bounded_stderr_tail(speech_child):
    body = "sys.stderr.write('discard-me' * 100000 + 'last diagnostic')\nsys.stderr.flush()\n"
    provider, processes, _ = speech_child(body + emit_script([ready(), final()]) + "sys.exit(7)\n")
    with pytest.raises(SpeechCaptureError) as caught:
        provider.capture_stream()
    diagnostic = caught.value.payload["stderr"]
    assert len(diagnostic.encode("utf-8")) <= 16 * 1024
    assert diagnostic.endswith("last diagnostic")
    assert_reaped(processes)


def test_duplicate_json_keys_cannot_rewrite_terminal_metadata(speech_child):
    ambiguous_final = json.dumps(final()).replace('"event": "final"', '"event": "error", "event": "final"')
    provider, processes, _ = speech_child(emit_script([ready()]) + f"print({ambiguous_final!r}, flush=True)\n")
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


def test_unterminated_line_is_capped_before_child_exit(speech_child):
    body = "sys.stdout.buffer.write(b'x' * 65537)\nsys.stdout.flush()\ntime.sleep(4)\n"
    provider, processes, _ = speech_child(body)
    started = time.monotonic()
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert time.monotonic() - started < 3, "The reader waited for EOF instead of limiting an unfinished line."
    assert_reaped(processes)


def test_total_stdout_cap_applies_to_individually_valid_lines(speech_child):
    events = [ready()] + [partial("x" * 60000, sequence=index + 2) for index in range(18)]
    provider, processes, _ = speech_child(emit_script(events + [final(sequence=20)]))
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


def test_event_count_cap_applies_to_small_valid_records(speech_child):
    events = [ready()] + [partial(sequence=index + 2) for index in range(1024)]
    provider, processes, _ = speech_child(emit_script(events + [final(sequence=1026)]))
    with pytest.raises(SpeechCaptureError):
        provider.capture_stream()
    assert_reaped(processes)


def test_timeout_terminates_even_a_child_that_ignores_terminate(speech_child, monkeypatch):
    shorten_deadline(monkeypatch)
    body = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + emit_script([ready(duration=0.05)]) + "time.sleep(30)\n"
    provider, processes, _ = speech_child(body)
    started = time.monotonic()
    with pytest.raises(SpeechCaptureError) as caught:
        provider.capture_stream(duration_s=0.05)
    assert caught.value.code == "capture_timeout"
    assert time.monotonic() - started < 3
    assert_reaped(processes)


def test_final_and_pipe_eof_still_require_successful_process_exit(speech_child, monkeypatch):
    shorten_deadline(monkeypatch)
    body = emit_script([ready(duration=0.05), final(duration=0.04, maximum=0.05)])
    body += "os.close(1)\nos.close(2)\ntime.sleep(30)\n"
    provider, processes, _ = speech_child(body)
    with pytest.raises(SpeechCaptureError) as caught:
        provider.capture_stream(duration_s=0.05)
    assert caught.value.code == "capture_timeout"
    assert_reaped(processes)


@pytest.mark.parametrize("failure", [RuntimeError("Display callback failed."), KeyboardInterrupt()])
def test_callback_failure_or_interrupt_reaps_child(speech_child, failure):
    provider, processes, _ = speech_child(emit_script([ready(), partial()]) + "time.sleep(30)\n")

    def fail_on_partial(event):
        if event.kind == "partial":
            raise failure

    with pytest.raises(type(failure)):
        provider.capture_stream(on_event=fail_on_partial)
    assert_reaped(processes)


@pytest.mark.parametrize("arguments", [
    {"duration_s": 0}, {"duration_s": float("nan")}, {"duration_s": 31},
    {"silence_duration_s": 0.1}, {"silence_duration_s": 5.1},
    {"silence_duration_s": float("nan")}, {"silence_duration_s": True},
    {"locale": "en-US\0"},
])
def test_invalid_stream_arguments_never_start_native_helper(speech_child, arguments):
    provider, processes, _ = speech_child(emit_script([ready(), final()]))
    with pytest.raises(SpeechCaptureError) as caught:
        provider.capture_stream(**arguments)
    assert caught.value.code == "invalid_capture_arguments"
    assert processes == []
