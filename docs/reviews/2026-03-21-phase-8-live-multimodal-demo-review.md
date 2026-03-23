# 2026-03-21 Phase 8 Live Multimodal Demo Review

## Scope

Review the Phase 8 patch for:

- live gaze-context freshness and grounding behavior
- live confirmation-turn correctness
- tester-facing CLI output and guidance quality

## Verification

- `pytest -q`
- `python scripts/phase5_productization_smoke.py fusion-smoke --runtime-dir /tmp/interaction-phase5-fusion-phase8`
- `python scripts/phase8_live_fusion_smoke.py --runtime-dir /tmp/interaction-phase8-live --gaze-frames 6 --duration 0.5 --confirm-duration 0.5`

## Findings

- No blocking findings.

## Notes

- The scripted `fusion-smoke` path still gives a deterministic regression harness for the fused confirmation flow.
- In this environment, the live Phase 8 smoke returns a structured `camera_unavailable` recovery payload once a placeholder calibration profile exists, which is the expected tester-facing behavior under denied macOS camera permission.
- OpenCV still emits camera-permission denial lines to stderr before the JSON payload. This remains documented in `docs/bugs.md` as a residual limitation rather than a blocker for the dry-run MVP.
