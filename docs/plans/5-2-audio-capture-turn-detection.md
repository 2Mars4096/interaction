# 5-2: Audio Capture and Turn Detection

**Parent:** [5-phase-2-voice-mvp](5-phase-2-voice-mvp.md)
**Status:** completed
**Goal:** Implement the input-turn pipeline needed to drive transcript streaming and push-to-talk command turns.

## Tasks
- [x] 1. Create the audio module boundary
  - [x] 1-1. Transcript-source abstraction via `ScriptedTranscriber`
  - [x] 1-2. Stream lifecycle management via `PushToTalkTurnManager`
- [x] 2. Implement turn segmentation
  - [x] 2-1. Push-to-talk turn boundary logic
  - [x] 2-2. Partial vs final segment emission
- [x] 3. Add resilience
  - [x] 3-1. Empty-input recovery path
  - [x] 3-2. Reset behavior
- [x] 4. Add tests or harnesses for stream and turn behavior

## Decisions
- The repository does not yet choose a microphone/STT vendor, so the first Phase 2 transport layer is transcript-driven rather than hardware-driven.
- Push-to-talk is represented explicitly as a turn manager instead of being hidden inside a UI layer.

## Notes
- Keep this slice transport-oriented. Do not bury intent logic in the audio layer.
