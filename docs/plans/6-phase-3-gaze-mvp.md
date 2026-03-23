# 6: Phase 3 Gaze MVP

**Status:** completed
**Goal:** Add webcam-based gaze targeting, calibration, and coarse trigger behavior that can support the later multimodal fusion phase.

## Tasks
- [x] 1. Finalize the phase spec and patch it before code
  - [x] 1-1. Complete [6-1-phase-3-spec-review](6-1-phase-3-spec-review.md)
- [x] 2. Build calibration and smoothing
  - [x] 2-1. Complete [6-2-calibration-smoothing](6-2-calibration-smoothing.md)
- [x] 3. Build target inference and trigger actions
  - [x] 3-1. Complete [6-3-target-inference-triggers](6-3-target-inference-triggers.md)
- [x] 4. Close the phase with a review gate
  - [x] 4-1. Complete [6-4-phase-3-code-review-gate](6-4-phase-3-code-review-gate.md)
- [ ] 5. Improve realism with a webcam-backed provider patch
  - [x] 5-1. Complete [6-5-webcam-gaze-provider-patch](6-5-webcam-gaze-provider-patch.md)
  - [x] 5-2. Complete [6-6-webcam-gaze-provider-review-gate](6-6-webcam-gaze-provider-review-gate.md)

## Decisions
- The webcam-first constraint means the phase optimizes for coarse, stable targeting before precision.
- Dwell is the default conservative trigger model for Phase 3.

## Notes
- Phase 3 should prepare reliable target grounding, not perfect mouse replacement.
- Phase 3 is now complete under the reviewed scope: provider-agnostic calibration, smoothing, large-target inference, conservative dwell triggering, broker/macOS dry-run integration, and a scripted gaze harness.
- The webcam-backed provider patch is a follow-on realism pass built on top of that completed scope.
- The webcam-backed provider patch is now also complete under its reviewed scope.
