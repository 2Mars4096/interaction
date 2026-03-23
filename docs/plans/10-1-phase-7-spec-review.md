# 10-1: Phase 7 Spec Review

**Parent:** [10-phase-7-webcam-mvp-completion](10-phase-7-webcam-mvp-completion.md)
**Status:** completed
**Goal:** Patch the webcam-MVP plan until live calibration scope, target model, and tester flow are concrete.

## Tasks
- [x] 1. Define the live webcam scope
  - [x] 1-1. Decide what counts as “complete” for the webcam path
  - [x] 1-2. Decide whether calibration is required in the first tester flow
- [x] 2. Define the target model
  - [x] 2-1. Choose coarse target regions for the first live run
  - [x] 2-2. Decide which actions the live webcam path should drive
- [x] 3. Define the tester flow
  - [x] 3-1. Decide how calibration is collected
  - [x] 3-2. Decide how failures are surfaced
- [x] 4. Patch roadmap docs if needed

## Decisions
- Webcam eye tracking will count as MVP-complete when there is a documented `gaze-calibrate` step, a persistent live calibration profile, a `gaze-live` tester path, and structured recovery for camera or detection failures.
- The first live webcam target model will use large coarse screen regions rather than pixel-precise controls.
- The first live webcam action surface will remain conservative: it should ground and highlight large targets through dwell, not attempt gaze-only clicking.
- Calibration should be collected through timed CLI prompts rather than an unbuilt GUI overlay so the first tester path remains runnable immediately.

## Notes
- This phase is not the place to pursue precision cursor control.
