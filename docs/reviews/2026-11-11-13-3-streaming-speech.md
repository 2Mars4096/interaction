# 13-3: Streaming Speech Review

**Code review:** passed
**Native distribution gate:** passed on macOS CI
**Complete CI matrix:** passed after the feedback fix
**Plan:** [13-3-streaming-speech](../plans/13-3-streaming-speech.md)

## Findings and fixes
- Independent reviews covered Python transport, CLI/final dispatch, native callback ownership, PCM indexing, endpointing, cancellation, and installed-wheel probes.
- Malformed array/object endpoint reasons originally raised an unstructured TypeError. Explicit type checking and regression cases now reject them as invalid helper output.
- Maximum-duration clipping is a failure, never an executable endAudio-generated partial final. Continuous sessions and bounded confirmation retries can ask for a repeat.
- Finalization's eight-second grace is measured from the capture endpoint; slow engine shutdown cannot extend it.
- Native recognition callbacks use the main queue, while authorization callbacks explicitly dispatch there. PCM energy uses each channel's sample stride, and the audio lock prevents append-after-endAudio races.
- Every emitted preview remains display/journal-only. Final text reaches the existing action/context checks exactly once, after successful protocol completion and helper exit.
- CI caught a late feedback-rendering regression: confirmation events carry final transcripts and were incorrectly classified as previews. The classifier now excludes decisions/proposals/results and preserves meaningful messages. Existing recorded-mode prompt checks and pre-capture assertions for streamed voice/fusion confirmations pass; independent review found no remaining issue.

## Evidence
- Full automated suite after the feedback fix: **623 passed** (the earlier implementation run had 621 tests).
- Sixty transport regressions, including real-child lifecycle tests, cover revisions, Unicode/dictation fidelity, strict metadata/order/framing, unknown confidence, bounded stderr/stdout/events, timeout, unsuccessful exit, callback failures, SIGTERM resistance, and process/pipe cleanup.
- CLI tests cover visible previews before capture completion, zero partial execution, streamed confirmations, errors after previews, changed focus, stop, capped retained history, and recorded/smoke compatibility.
- Local sdist-to-wheel build, packaged resource checks, critical Ruff, and `git diff --check` passed.
- [Initial CI run](https://github.com/2Mars4096/interaction/actions/runs/42993369577): both distribution jobs passed, including installed-wheel native compilation and four pre-permission argument probes on macOS. The six unit-test jobs exposed the confirmation-rendering regression above.
- [Corrected CI run](https://github.com/2Mars4096/interaction/actions/runs/42994103807) at `5e6f197`: all eight jobs passed. The six Linux/macOS Python 3.11–3.13 test jobs passed with the 623-test suite; both distribution jobs passed, including installed-wheel native compilation and pre-permission probes on macOS.

## Limits
- No microphone/camera evaluation occurred. Silence thresholds, endpoint accuracy, transcription accuracy, and measured end-to-end latency require the user's Mac.
- Streaming is opt-in; recorded capture remains the compatibility default.
- Speech is streamed within each turn. Native app UI, semantic Accessibility targets, wake words, always-open audio, and action barge-in remain separate milestones.
