# 12-4: Tracking Recovery and Deliberate Dwell

**Parent:** [12-reliable-live-interaction](12-reliable-live-interaction.md)
**Status:** completed
**Goal:** Require sustained current gaze evidence and prevent stale dwell or drag state from producing accidental actions.

## Tasks
- [x] Reset smoothing, partial dwell, and armed drag on lost/weak tracking or long sampling gaps.
- [x] Keep a fired dwell target disarmed until observed gaze deliberately leaves it; tracking loss alone must not rearm it.
- [x] Measure live timing from a monotonic clock and pace captures explicitly rather than crediting nominal frame duration.
- [x] Share stable-target evidence across live fusion and bounded context capture: sample count, duration, confidence, point dispersion, and maximum gap.
- [x] Remove target-confidence inflation; identify detector evidence separately from calibration accuracy in docs/output.
- [x] Cover one-frame locks, intermittent loss, long pauses, same-target repeats, drag cancellation, and recovery with deterministic tests.
- [x] Review, document behavior, commit, and push.

## Decisions and plan review
- Coarse screen regions remain the target model; no Accessibility target snapping or tracker swap in this section.
- A fired target stays disarmed across dropped frames. Rearming requires a sustained valid look away, then a new complete dwell.
- Lost tracking cancels an armed drag before another sample can execute its destination.
- Scripted traces can still pass explicit sample durations. Live helpers derive time from the clock and never count a stalled camera read as fixation.
- Default stable lock evidence is at least three samples over 120 ms, with a bounded gap and normalized-point radius. These are explicit engineering gates, not a calibrated accuracy probability.
