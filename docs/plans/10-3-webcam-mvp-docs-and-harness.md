# 10-3: Webcam MVP Docs and Harness

**Parent:** [10-phase-7-webcam-mvp-completion](10-phase-7-webcam-mvp-completion.md)
**Status:** completed
**Goal:** Update the tester guidance, MVP criteria, and smoke commands so live webcam gaze is part of the declared MVP path.

## Tasks
- [x] 1. Update readiness targets for webcam eye tracking
  - [x] 1-1. Calibration expectations
  - [x] 1-2. Live run expectations
- [x] 2. Add tester-facing harness guidance
  - [x] 2-1. Document calibration and live-run commands
  - [x] 2-2. Document camera-permission and no-detection failure modes
- [x] 3. Update roadmap and readme language for the webcam-complete MVP state

## Decisions
- `docs/mvp-readiness.md` now treats live webcam calibration and `gaze-live` as part of the recommended MVP tester flow, not an optional low-level diagnostic.
- The tester guidance now assumes calibration should happen before live webcam gaze runs.
- The old standalone Phase 3 webcam smoke remains useful as a low-level diagnostic, but it is no longer the primary tester-facing webcam path.

## Notes
- This subplan should make it obvious to the user how to try live webcam gaze without reading the implementation.
