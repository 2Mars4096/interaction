# 11-3: Live Fusion Docs and Harness

**Parent:** [11-phase-8-live-multimodal-demo](11-phase-8-live-multimodal-demo.md)
**Status:** completed
**Goal:** Update the tester guidance and wrappers so the live multimodal demo path is explicit and easy to run.

## Tasks
- [x] 1. Add a repo-local live fusion smoke wrapper
- [x] 2. Update MVP readiness and README guidance
- [x] 3. Clarify the sequential live capture model and its limitations

## Decisions
- The repo-local tester wrapper is `python scripts/phase8_live_fusion_smoke.py`.
- The tester docs should describe `fusion-live` as a real sequential gaze-then-voice demo, not a continuous concurrent session product.
- The docs should keep the scripted `fusion-smoke` path alongside `fusion-live` so regression testing stays deterministic.

## Notes
- The docs should make clear that this is a real live demo, but not yet a continuous session product.
