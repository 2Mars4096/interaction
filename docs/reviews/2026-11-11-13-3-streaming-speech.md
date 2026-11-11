# 13-3: Streaming Speech Review

**Code review:** passed
**Native distribution gate:** pending macOS CI
**Plan:** [13-3-streaming-speech](../plans/13-3-streaming-speech.md)

## Findings and fixes
- Independent reviews covered Python transport, CLI/final dispatch, native callback ownership, PCM indexing, endpointing, cancellation, and installed-wheel probes.
- Malformed array/object endpoint reasons originally raised an unstructured TypeError. Explicit type checking and regression cases now reject them as invalid helper output.
- Maximum-duration clipping is a failure, never an executable endAudio-generated partial final. Continuous sessions and bounded confirmation retries can ask for a repeat.
- Finalization's eight-second grace is measured from the capture endpoint; slow engine shutdown cannot extend it.
- Native recognition callbacks use the main queue, while authorization callbacks explicitly dispatch there. PCM energy uses each channel's sample stride, and the audio lock prevents append-after-endAudio races.
- Every emitted preview remains display/journal-only. Final text reaches the existing action/context checks exactly once, after successful protocol completion and helper exit.

## Evidence
- Full automated suite: **621 passed**.
- Sixty real-child transport tests cover revisions, Unicode/dictation fidelity, strict metadata/order/framing, unknown confidence, bounded stderr/stdout/events, timeout, unsuccessful exit, callback failures, SIGTERM resistance, and process/pipe cleanup.
- CLI tests cover visible previews before capture completion, zero partial execution, streamed confirmations, errors after previews, changed focus, stop, capped retained history, and recorded/smoke compatibility.
- Local sdist-to-wheel build, packaged resource checks, critical Ruff, and `git diff --check` passed.
- Installed-wheel macOS verification compiles the packaged source and probes four invalid argument cases before any permission or capture API. Its execution is pending CI.

## Limits
- No microphone/camera evaluation occurred. Silence thresholds, endpoint accuracy, transcription accuracy, and measured end-to-end latency require the user's Mac.
- Streaming is opt-in; recorded capture remains the compatibility default.
- Speech is streamed within each turn. Native app UI, semantic Accessibility targets, wake words, always-open audio, and action barge-in remain separate milestones.
