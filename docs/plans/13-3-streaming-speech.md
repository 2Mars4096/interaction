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
- [ ] Compile packaged native resources and run pre-permission argument probes on macOS CI.

## Decisions
- Never promote a partial transcript after timeout or recognition error. Silence requests finalization; it does not authorize partial text.
- Keep microphone/speech permission prompts and existing source/OS-specific helper caching.
- Maximum capture duration is a ceiling, not an imposed wait in streaming mode. Recognition finalization has a separate bounded grace period.
- This is streaming within each turn, not an always-open microphone daemon, wake-word engine, or barge-in during synchronous desktop execution.
- Protocol, provider callback, CLI options, numeric/output limits, and source references are specified in [speech-stream-protocol](../speech-stream-protocol.md).

## Verification
- 621 tests passed locally, including 60 real-child transport cases and live-session streaming integration checks.
- Independent Python/CLI/native code review passed after fixing malformed endpoint metadata handling. Native finalization is bounded from the actual capture endpoint.
- Native compilation and installed-helper argument probes are the remaining CI gate. Acoustic latency/accuracy evaluation remains separate.
