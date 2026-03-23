# 11-2: Live Fusion CLI and Runtime

**Parent:** [11-phase-8-live-multimodal-demo](11-phase-8-live-multimodal-demo.md)
**Status:** completed
**Goal:** Build the first real live multimodal command path on top of the saved webcam calibration and live speech provider.

## Tasks
- [x] 1. Add a live gaze-context capture helper
  - [x] 1-1. Convert live webcam readings into a fresh grounded target
  - [x] 1-2. Report missing or low-confidence gaze context clearly
- [x] 2. Add the `fusion-live` CLI path
  - [x] 2-1. Capture live gaze context
  - [x] 2-2. Capture live voice
  - [x] 2-3. Capture an optional live confirmation turn
- [x] 3. Add tests for success and failure paths
  - [x] 3-1. missing calibration or camera
  - [x] 3-2. missing microphone or speech permission
  - [x] 3-3. confirmation-required live flow

## Decisions
- The live gaze-context helper should reuse the saved `webcam-live` calibration profile and the existing 3x3 large-region target model.
- The grounded live gaze target should be the dominant inferred target across the short capture window, with the freshest matching observation attached to fusion state.
- The CLI should only capture a second live voice turn if the first fused command leaves a pending confirmation decision.
- Missing calibration, missing grounded gaze, camera errors, and speech errors should all return structured payloads instead of exceptions.
- The first implementation keeps live gaze and live speech sequential rather than concurrent so the MVP remains simple to test and review.

## Notes
- This path should remain dry-run by default so live multimodal testing is safe.
