# 10-2: Live Webcam Calibration and CLI

**Parent:** [10-phase-7-webcam-mvp-completion](10-phase-7-webcam-mvp-completion.md)
**Status:** completed
**Goal:** Add the live webcam calibration flow, persistence, and CLI command surface needed for a tester-facing gaze path.

## Tasks
- [x] 1. Improve the OpenCV provider surface
  - [x] 1-1. Add structured camera-open failure handling
  - [x] 1-2. Add helper routines for averaged webcam samples
- [x] 2. Add calibration orchestration
  - [x] 2-1. Collect calibration samples through timed prompts
  - [x] 2-2. Persist the resulting live webcam calibration profile
- [x] 3. Add the live webcam CLI path
  - [x] 3-1. Add `gaze-live`
  - [x] 3-2. Add `gaze-calibrate`
- [x] 4. Add tests for calibration capture, CLI success paths, and structured failure paths

## Decisions
- The OpenCV provider now exposes structured camera-open failures and averaged sample capture, which is enough for a CLI-driven calibration flow.
- Live webcam calibration is persisted under the `webcam-live` profile name and is now required by default before `gaze-live` will run.
- The first live webcam CLI path uses large 3x3 screen regions and conservative dwell-triggered highlighting rather than pointer movement or clicking.

## Notes
- This subplan is where webcam gaze becomes a product surface instead of just a CV helper.
