# 7-1: Phase 4 Spec Review

**Parent:** [7-phase-4-multimodal-fusion](7-phase-4-multimodal-fusion.md)
**Status:** completed
**Goal:** Patch the fusion-phase plan until shared-state timing, grounding windows, and clarification policy are explicit.

## Tasks
- [x] 1. Define fusion timing rules
  - [x] 1-1. How long gaze context remains valid for a spoken command
  - [x] 1-2. How recent transcript and target history are used
- [x] 2. Define ambiguity policy
  - [x] 2-1. When to clarify
  - [x] 2-2. When to highlight multiple options
- [x] 3. Patch contracts and docs if needed
- [x] 4. Freeze the Phase 4 acceptance criteria

## Decisions
- A grounded gaze target is valid for deictic voice commands for a short window after the last stable fixation; the initial default is about 1.5 seconds.
- Fusion should prefer the latest stable grounded target, but it may surface a small candidate list for clarification when recent targets compete.
- Deictic commands such as `click this`, `open this`, `focus here`, and `show this` should resolve against recent gaze context instead of auto-clarifying.
- If target confidence is too low, target history is stale, or recent targets conflict materially, the system should clarify rather than guess.
- Phase 4 acceptance is defined around explicit shared state, reproducible fused command handling, confidence aggregation, clarification behavior, and repeatable evaluation output.

## Notes
- The plan review should make temporal assumptions explicit before code.
