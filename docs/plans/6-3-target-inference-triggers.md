# 6-3: Target Inference and Trigger Actions

**Parent:** [6-phase-3-gaze-mvp](6-phase-3-gaze-mvp.md)
**Status:** completed
**Goal:** Turn smoothed gaze signals into grounded targets and conservative trigger behavior.

## Tasks
- [x] 1. Implement coarse target inference
  - [x] 1-1. Map gaze to regions or candidate targets
  - [x] 1-2. Rank likely targets with confidence
- [x] 2. Implement conservative trigger behavior
  - [x] 2-1. Highlight target on stable fixation
  - [x] 2-2. Add dwell or eye-gesture trigger for low-risk actions
- [x] 3. Wire gaze outputs into the contract layer
  - [x] 3-1. Emit grounded targets
  - [x] 3-2. Emit gaze observations compatible with later fusion
- [x] 4. Add tests and manual evaluation harnesses

## Decisions
- Large-target containment plus center-distance scoring is sufficient for the first coarse inferencer.
- Dwell triggers `highlight_target` by default; clicks remain deferred to later multimodal grounding rather than gaze-only activation.

## Notes
- This phase should prefer false negatives over false positive clicks.
