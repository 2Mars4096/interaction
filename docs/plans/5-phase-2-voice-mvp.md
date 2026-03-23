# 5: Phase 2 Voice MVP

**Status:** completed
**Goal:** Add a usable realtime voice loop on top of the broker and macOS control base.

## Tasks
- [x] 1. Finalize the phase spec and patch it before code
  - [x] 1-1. Complete [5-1-phase-2-spec-review](5-1-phase-2-spec-review.md)
- [x] 2. Build audio capture and turn handling
  - [x] 2-1. Complete [5-2-audio-capture-turn-detection](5-2-audio-capture-turn-detection.md)
- [x] 3. Build transcription and voice-intent flow
  - [x] 3-1. Complete [5-3-transcription-intent-loop](5-3-transcription-intent-loop.md)
- [x] 4. Build feedback, interruption, and confirmation UX
  - [x] 4-1. Complete [5-4-feedback-confirmation-ux](5-4-feedback-confirmation-ux.md)
- [x] 5. Close the phase with a review gate
  - [x] 5-1. Complete [5-5-phase-2-code-review-gate](5-5-phase-2-code-review-gate.md)

## Decisions
- Voice should reach a useful standalone loop before gaze is added.
- Push-to-talk is the initial interaction mode for Phase 2.

## Notes
- Phase 2 should still be useful in a voice-only mode for commands that do not require precise target grounding.
- Phase 2 is now complete under the reviewed scope: push-to-talk transcript streaming, bounded voice parsing, confirmation/cancellation, broker integration, and dry-run macOS execution.
