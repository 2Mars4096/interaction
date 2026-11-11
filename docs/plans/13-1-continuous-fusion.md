# 13-1: Continuous Fusion

**Parent:** [13-responsive-live-sessions](13-responsive-live-sessions.md)
**Status:** completed
**Goal:** Accept successive voice-and-gaze commands without restarting the CLI or camera.

## Tasks
- [x] Add `--turns`, `--continuous`, and bounded `--confirmation-attempts` to `fusion-live`.
- [x] Keep one camera owner and one fusion loop for the whole session; refresh gaze after every capture.
- [x] Recover from silence and changed foreground context in continuous mode; retry unclear confirmation within its budget, then cancel.
- [x] Handle spoken stop and Ctrl-C, cancel pending work, and close the camera exactly once on all exits.
- [x] Preserve existing single-command payload fields and add capture history and explicit session outcome.
- [x] Test repeated commands, confirmation/cancel/stop, context and tracking changes, error recovery, immediate feedback, and cleanup; review before commit/push.

## Decisions
- Preserve the initial calibration and stable-gaze startup gate. Subsequent non-pointer commands and stop remain usable after tracking loss.
- Do not repeatedly call the sampler's startup-only frame-budget wait; per-command gaze comes from its fresh snapshot.
- Clear stale gaze error fields when tracking recovers. Detector confidence remains a heuristic.
- Keep all action confirmation policy unchanged; coarse screen regions do not identify semantic controls.
- Honor a captured session stop before action-context validation in voice and fusion; changing or unknown foreground context cannot discard the stop.

## Verification
- Independent implementation review found no blocking issues.
- 509 tests passed; critical Ruff checks and `git diff --check` passed.
- Review: [13-1 continuous fusion](../reviews/2026-11-11-13-1-continuous-fusion.md).
