# 5-4: Feedback, Confirmation, and Interruption UX

**Parent:** [5-phase-2-voice-mvp](5-phase-2-voice-mvp.md)
**Status:** completed
**Goal:** Add the user-facing loop that makes the voice system understandable, interruptible, and safe.

## Tasks
- [x] 1. Implement transcript feedback
  - [x] 1-1. Show partial transcript
  - [x] 1-2. Show final interpreted command
- [x] 2. Implement confirmation UX
  - [x] 2-1. Prompt for confirm decisions
  - [x] 2-2. Route spoken confirmation and rejection
- [x] 3. Implement interruption and cancellation
  - [x] 3-1. Stop pending execution
  - [x] 3-2. Reset session state safely
- [x] 4. Add tests or manual harnesses for user-visible state transitions

## Decisions
- Phase 2 feedback is event-driven textual feedback rather than a polished overlay.
- Voice confirmation and cancellation are handled as normal utterances when a pending confirmation exists.

## Notes
- The product will feel unreliable if this UX is weak even when transcription quality is decent.
