# 13-3: Streaming Speech

**Parent:** [13-responsive-live-sessions](13-responsive-live-sessions.md)
**Status:** in-progress
**Goal:** Recognize microphone audio while speaking and end a turn after a bounded silence interval, without executing partial transcripts.

## Tasks
- [x] Add a versioned native JSONL speech-event protocol and bounded Python subprocess reader with cancellation/timeout cleanup.
- [x] Add an AVAudioEngine/SFSpeechAudioBufferRecognitionRequest capture path with partial previews, a maximum turn duration, and explicit speech endpointing.
- [x] Require a valid final result and successful helper exit before returning executable text; fail closed on errors, malformed/out-of-order events, and missing finalization.
- [x] Expose streaming mode and silence interval to live voice/fusion commands while preserving recorded capture as the compatibility default.
- [x] Show real partial transcripts immediately on stderr and record them through the existing live event journal; no partial enters the broker.
- [x] Exercise real harmless-child protocol/lifecycle failures and CLI integration; record hardware checks separately.
- [x] Compile packaged native resources and run pre-permission argument probes on macOS CI.
- [ ] Pass the complete CI matrix after correcting confirmation feedback identified by the first run.

## Decisions
- Never promote a partial transcript after timeout or recognition error. Silence requests finalization; it does not authorize partial text.
- Keep microphone/speech permission prompts and existing source/OS-specific helper caching.
- Maximum capture duration is a ceiling, not an imposed wait in streaming mode. Recognition finalization has a separate bounded grace period.
- This is streaming within each turn, not an always-open microphone daemon, wake-word engine, or barge-in during synchronous desktop execution.
- Protocol, provider callback, CLI options, numeric/output limits, and source references are specified in [speech-stream-protocol](../speech-stream-protocol.md).

## Verification
- 623 tests passed locally after the confirmation-feedback fix, including 60 transport regressions and live-session streaming integration checks.
- Independent Python/CLI/native code review passed after fixing malformed endpoint metadata handling. Native finalization is bounded from the actual capture endpoint.
- Native compilation and installed-helper argument probes passed in [CI](https://github.com/2Mars4096/interaction/actions/runs/42993369577). The complete matrix must pass with the confirmation-feedback fix before closure. Acoustic latency/accuracy evaluation remains separate.
