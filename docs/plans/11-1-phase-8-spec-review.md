# 11-1: Phase 8 Spec Review

**Parent:** [11-phase-8-live-multimodal-demo](11-phase-8-live-multimodal-demo.md)
**Status:** completed
**Goal:** Patch the live-multimodal plan until the capture sequence, confirmation flow, and tester expectations are concrete.

## Tasks
- [x] 1. Define the live multimodal sequence
  - [x] 1-1. Decide whether gaze and voice are captured simultaneously or sequentially
  - [x] 1-2. Decide how confirmation works for medium-risk actions
- [x] 2. Define the live grounding model
  - [x] 2-1. Decide how gaze context is sourced
  - [x] 2-2. Decide how stale or missing gaze should fail
- [x] 3. Define the tester flow
  - [x] 3-1. Decide the CLI command shape
  - [x] 3-2. Decide what output the command must provide
- [x] 4. Patch roadmap docs if needed

## Decisions
- The first live multimodal run will be sequential: capture a short live gaze context window first, then capture live speech, then optionally capture a second live speech confirmation turn if the broker requires it.
- Live gaze context will come from the saved `webcam-live` calibration profile and the existing large-region webcam target model.
- Missing calibration, camera denial, missing gaze lock, microphone denial, and speech denial must all fail with structured recovery output.
- The tester-facing command will be `interaction fusion-live`, and it must log both the capture metadata and the fusion event sequence.

## Notes
- Full continuous gaze-plus-audio concurrency is out of scope for this phase.
