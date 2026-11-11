# Section 4: Tracking Recovery Review

**Gate:** passed after fixes
**Plan:** [12-4-tracking-recovery](../plans/12-4-tracking-recovery.md)

## Findings and fixes
- The first frame on another target could incorrectly count the preceding interval as look-away and rearm a fired target. Live measured timing now credits zero fixation on first/transition samples while independently aging cooldown and drag timeouts.
- Ending a bounded live session could leave a drag armed in a reused loop. Completion, provider errors, and interruption now clear partial dwell, smoothing, and armed drag in a finally path.

## Evidence
- Full suite: 389 passed; `git diff --check` passed.
- Regression reproduces the original false rearm with A x5, B x1, A x4 at 200 ms per read; only one action now occurs.
- Stable evidence requires count and actual duration, finite in-bounds unsmoothed coordinates, bounded displacement, and bounded sampling gaps.
- Tests cover dropped/weak frames, jitter, slow drift, long read stalls, stale final bounded context, snapshot expiry, cursor reacquisition, and loss/changed-context drag cancellation.
- Fired-target latches survive cooldown and tracking loss; sustained valid look-away and a complete new dwell are tested independently.
- Terminal gaze feedback is delivered immediately and shows dwell progress/rearm state.

## Limits
- Confidence is a detector heuristic, distinct from saved calibration quality.
- Primary-display coarse regions remain the target model. Actual targeting performance, native feedback behavior, and Accessibility-based control identity still require further work and hardware checks.
