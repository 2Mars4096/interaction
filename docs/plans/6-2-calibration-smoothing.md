# 6-2: Calibration and Smoothing

**Parent:** [6-phase-3-gaze-mvp](6-phase-3-gaze-mvp.md)
**Status:** completed
**Goal:** Implement calibration, smoothing, and stability logic so gaze can support coarse target grounding.

## Tasks
- [x] 1. Implement calibration flow
  - [x] 1-1. Calibration targets and routine
  - [x] 1-2. Calibration result persistence for the session
- [x] 2. Implement smoothing and stability logic
  - [x] 2-1. Jitter reduction
  - [x] 2-2. Stable fixation detection
- [x] 3. Add dropout handling
  - [x] 3-1. No-target / low-grounding recovery behavior
  - [x] 3-2. Recalibration-ready profile boundary
- [x] 4. Add tests or measurement harnesses for stability behavior

## Decisions
- Calibration uses a lightweight affine fit over normalized points so the pipeline stays provider-agnostic.
- Smoothing is a simple moving average tuned for coarse stability, not high-frequency precision.

## Notes
- This subplan is about making gaze signals usable, not yet about rich intent.
