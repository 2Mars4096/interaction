# 9-3: MVP Readiness and Live Harness

**Parent:** [9-phase-6-mvp-readiness](9-phase-6-mvp-readiness.md)
**Status:** completed
**Goal:** Define the first tester-ready acceptance targets and expose a live harness the user can run when the MVP is ready.

## Tasks
- [x] 1. Define readiness targets
  - [x] 1-1. Latency target
  - [x] 1-2. Accuracy and accidental-action target
- [x] 2. Add the tester-facing harness flow
  - [x] 2-1. Add CLI or smoke commands for live voice testing
  - [x] 2-2. Document permission setup and failure modes
- [x] 3. Update roadmap and readme guidance for the MVP-ready state

## Decisions
- The first MVP tester flow is now voice-first: live macOS microphone transcription plus the existing broker, adapter, overlay, and logging stack.
- `docs/mvp-readiness.md` now records explicit latency, accuracy, accidental-action, and permission-handling targets for the first user trial.
- The backlog item for acceptable latency, accuracy, and accidental-action targets is now closed.

## Notes
- This subplan should close the backlog item around acceptable latency, accuracy, and accidental-action targets.
