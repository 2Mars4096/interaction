# 6-1: Phase 3 Spec Review

**Parent:** [6-phase-3-gaze-mvp](6-phase-3-gaze-mvp.md)
**Status:** completed
**Goal:** Patch the gaze-phase plan until calibration UX, target model assumptions, and trigger rules are concrete enough to build.

## Tasks
- [x] 1. Finalize the webcam tracking assumptions
  - [x] 1-1. Define expected seating and camera conditions
  - [x] 1-2. Define acceptable target sizes for MVP
- [x] 2. Finalize the trigger model
  - [x] 2-1. Decide dwell vs blink vs other gestures
  - [x] 2-2. Define conservative defaults
- [x] 3. Patch contracts and docs if needed
- [x] 4. Freeze the Phase 3 acceptance criteria

## Decisions
- Phase 3 assumes a normal seated desktop user facing a webcam in stable indoor lighting.
- The MVP gaze target model is for large targets and coarse regions, not tiny controls.
- Dwell is the default trigger model for Phase 3; blink- or facial-gesture triggers are deferred.
- Conservative defaults matter more than cleverness: stable dwell should highlight first, not click by default.
- Phase 3 acceptance is defined around calibration, smoothing, target inference, dwell-trigger behavior, and a provider-agnostic scripted gaze harness rather than a locked webcam vendor stack.

## Notes
- The review should explicitly avoid promising precision that a standard webcam cannot reliably deliver.
