# 13-3: Streaming Speech

**Parent:** [13-responsive-live-sessions](13-responsive-live-sessions.md)
**Status:** not-started
**Goal:** Recognize microphone audio while speaking and end a turn after a bounded silence interval, without executing partial transcripts.

## Tasks
- [ ] Add a versioned native JSONL speech-event protocol and bounded Python subprocess reader with cancellation/timeout cleanup.
- [ ] Add an AVAudioEngine/SFSpeechAudioBufferRecognitionRequest capture path with partial previews, a maximum turn duration, and explicit speech endpointing.
- [ ] Require a valid final result and successful helper exit before returning executable text; fail closed on errors, malformed/out-of-order events, and missing finalization.
- [ ] Expose streaming mode and silence interval to live voice/fusion commands while preserving recorded capture as the compatibility default.
- [ ] Show real partial transcripts immediately on stderr and record them through the existing live event journal; no partial enters the broker.
- [ ] Exercise fake-process protocol/lifecycle failures and CLI integration; compile packaged native resources on macOS CI; record hardware checks separately.

## Decisions
- Never promote a partial transcript after timeout or recognition error. Silence requests finalization; it does not authorize partial text.
- Keep microphone/speech permission prompts and existing source/OS-specific helper caching.
- Maximum capture duration is a ceiling, not an imposed wait in streaming mode. Recognition finalization has a separate bounded grace period.
- This is streaming within each turn, not an always-open microphone daemon, wake-word engine, or barge-in during synchronous desktop execution.
