# 6-5: Webcam Gaze Provider Patch

**Parent:** [6-phase-3-gaze-mvp](6-phase-3-gaze-mvp.md)
**Status:** completed
**Goal:** Add a real webcam-backed gaze provider path using locally available OpenCV primitives so Phase 3 can be tested beyond scripted traces.

## Tasks
- [x] 1. Review the patch scope before code
  - [x] 1-1. Keep the provider bounded to coarse gaze estimates for large targets
  - [x] 1-2. Avoid claiming precision beyond what Haar- and pupil-based heuristics can support
- [x] 2. Implement a webcam gaze provider
  - [x] 2-1. Capture frames from a local webcam with OpenCV
  - [x] 2-2. Detect face and eyes with OpenCV cascades
  - [x] 2-3. Estimate a coarse gaze point from detected eye regions
- [x] 3. Integrate the provider into the Phase 3 runtime
  - [x] 3-1. Convert webcam samples into existing gaze contracts
  - [x] 3-2. Add a manual smoke harness for live webcam testing
- [x] 4. Add tests and close with a review gate
  - [x] 4-1. Add unit tests for the provider math/helpers
  - [x] 4-2. Complete [6-6-webcam-gaze-provider-review-gate](6-6-webcam-gaze-provider-review-gate.md)

## Decisions
- This is a realism patch, not a replacement for the scripted Phase 3 scope already completed.
- The provider is allowed to be coarse and heuristic. The goal is “real webcam-backed large-target grounding,” not precision eye tracking.

## Notes
- OpenCV is available locally; this patch should avoid adding a hard dependency on MediaPipe or external download steps.
- Live camera access is environment-dependent; the smoke harness now reports permission failures cleanly.
