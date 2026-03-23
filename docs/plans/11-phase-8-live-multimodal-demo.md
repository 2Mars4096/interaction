# 11: Phase 8 Live Multimodal Demo

**Status:** completed
**Goal:** Add a tester-facing live multimodal command path that combines saved webcam gaze context with live macOS speech capture and the existing fusion broker.

## Tasks
- [x] 1. Finalize the phase spec and patch it before code
  - [x] 1-1. Complete [11-1-phase-8-spec-review](11-1-phase-8-spec-review.md)
- [x] 2. Build the live multimodal orchestration path
  - [x] 2-1. Complete [11-2-live-fusion-cli-and-runtime](11-2-live-fusion-cli-and-runtime.md)
- [x] 3. Update tester docs and MVP guidance
  - [x] 3-1. Complete [11-3-live-fusion-docs-and-harness](11-3-live-fusion-docs-and-harness.md)
- [x] 4. Close the phase with a review gate
  - [x] 4-1. Complete [11-4-phase-8-code-review-gate](11-4-phase-8-code-review-gate.md)

## Decisions
- The first live multimodal path should reuse the existing live voice helper and saved webcam calibration rather than invent a parallel stack.
- The first live demo can be sequential rather than fully simultaneous as long as gaze context is fresh and the end-to-end behavior is real.
- Phase 8 closes only when both the no-calibration path and the permission-sensitive camera path return structured tester-facing payloads instead of tracebacks.

## Notes
- This phase is about a coherent live demo path, not yet about continuous multimodal session management.
