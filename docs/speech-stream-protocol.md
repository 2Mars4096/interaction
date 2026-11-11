# Native Speech Stream Protocol (version 1)

The optional `--speech-mode stream` path invokes the packaged helper with
`--stream --duration MAX --locale LOCALE --silence-duration SECONDS`.
Recorded capture retains its existing arguments and one-document output.

## Event envelope

Every stdout line is one UTF-8 JSON object with `schema_version: 1`, a strictly
increasing contiguous integer `sequence` starting at 1, `provider: "macos_speech"`,
`locale`, and `event`. Only protocol records belong on stdout; diagnostics use
stderr. Flush each event immediately.

| Event | Required additional fields | Meaning |
|---|---|---|
| `ready` | `max_duration_s`, `silence_duration_s`, `permission_state: "granted"`, `used_on_device` (boolean) | Audio engine started; first non-error event, exactly once |
| `partial` | `transcript` (nonempty text), `elapsed_s` (finite nonnegative) | Revisable display-only recognition; zero or more after ready |
| `final` | `transcript`, `confidence` (null or finite 0–1), `duration_s`, `max_duration_s`, `endpoint_reason`, `permission_state: "granted"`, `used_on_device` | Exactly one terminal success, only after actual final recognition |
| `error` | `error` (nonempty code), `message` (nonempty text) | Terminal failure, permitted before ready |

Successful `endpoint_reason` is `silence` or `recognizer_final`.
Reaching the capture ceiling while speech is still active returns
`capture_limit_reached` with `endpoint_reason: "max_duration"` as error metadata;
it never makes clipped text executable. Continuous sessions can ask for a repeat.
`duration_s` measures capture time after engine startup; it may be shorter than
the requested maximum. A scheduling tolerance of 0.5 seconds above the maximum
is allowed. Partial elapsed time is monotonic and cannot exceed that capture
bound plus the separate 8-second recognition finalization budget.

## Turn completion and limits

- Stream microphone buffers to `SFSpeechAudioBufferRecognitionRequest` while
  reporting partial results. No temporary recording is required in stream mode.
- Use audio energy to detect silence after observed speech, rather than treating
  a stalled partial transcript as silence. Default silence interval is 0.8 seconds
  (configurable 0.2–5 seconds); initial energy threshold is -45 dBFS and requires
  at least 0.15 seconds of voiced audio. These are unmeasured engineering defaults.
- Stop the engine/remove its tap and call `endAudio` at the endpoint. Wait at most
  8 further seconds for an actual final result. A final callback with an error
  remains a failure. A timeout never promotes the last partial to final.
- The maximum capture duration is finite, positive, and at most 30 seconds.
- Python validates event ordering, metadata, numeric values, and a successful
  process exit before returning executable text. Duplicate terminal records,
  missing final output, nonzero exit, or malformed output produce no action.
- Bound each line to 64 KiB, total stdout to 1 MiB, total events to 1024, and
  retained stderr diagnostics to 16 KiB. Drain stderr concurrently.
- Bound the whole child lifetime, terminate/reap it on timeout, Ctrl-C, or callback
  failure, and close all pipes. Native capture releases the engine, tap, and
  recognition task on normal/error exits.

## Python and CLI contract

- `MacOSLiveSpeechProvider.capture_stream(duration_s=4.0, locale="en-US",
  silence_duration_s=0.8, on_event=None)` returns `LiveSpeechCapture` only after
  final validation and process exit. `capture_turn` remains recorded mode.
- Display callbacks receive `SpeechStreamEvent(kind, transcript, elapsed_s)` for
  `ready` and `partial` only. They never enter the command broker.
- A streamed capture includes `capture_mode: "stream"`, measured `duration_s`,
  `max_duration_s`, and `endpoint_reason`. Recorded captures use
  `capture_mode: "recorded"` and leave optional endpoint metadata unknown.
- CLI flags are `--speech-mode recorded|stream` (default recorded) and
  `--silence-duration` (default 0.8). Both command and confirmation captures use
  the selected mode. No automatic fallback repeats a failed turn.
- Live final text bypasses scripted word-prefix generation. The deterministic
  smoke paths retain their scripted transcript behavior.

## API references

- [Apple: live audio buffer recognition](https://developer.apple.com/documentation/speech/sfspeechaudiobufferrecognitionrequest)
- [Apple: partial result configuration](https://developer.apple.com/documentation/speech/sfspeechrecognitionrequest/shouldreportpartialresults)
- [Apple: audio tap lifecycle](https://developer.apple.com/documentation/avfaudio/avaudionode/installtap(onbus:buffersize:format:block:))

The current Objective-C helper targets the existing macOS support range; replacing
it with newer SpeechAnalyzer APIs is outside this section.
