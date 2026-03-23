# Changelog

## 2026-03-23
- [feat] Added normalized gaze-point execution planning so `fusion-live` can turn recent gaze context into real pointer `focus`, `click`, `double-click`, and `right-click` plans instead of placeholder notifications.
- [feat] Added `right_click_target` plus deictic multimodal support for phrases like `right click this` and `move here`.
- [feat] Added a one-shot `--execute` CLI override so live voice, gaze, and fusion commands can bypass the default dry-run setting for demos.
- [feat] Added explicit `gaze-live --action ...` dwell modes for gaze-only `move`, `click`, `right-click`, and `double-click` testing.
- [feat] Added bounded two-stage gaze-only drag support: dwell once to arm the drag origin, dwell again to drop, with a normalized `drag_target` execution plan underneath.
- [feat] Added continuous `gaze-live --action cursor` follow mode so the pointer can track calibrated gaze without waiting for a dwell trigger.
- [feat] Added cursor-follow tuning controls for deadzone, smoothing, edge padding, and max step so live cursor behavior can be adjusted per machine.
- [feat] Added gaze-action cooldown and drag timeout controls so live dwell sessions can suppress accidental repeat actions and recover from abandoned drag arms.
- [fix] Stopped drag mode from entering cooldown at the arm stage; cooldown now applies after the completed drop instead of blocking the second dwell.
- [infra] Added `numpy` and `opencv-python` to the installable project dependencies and ignored runtime state under `.interaction/`.
- [test] Added regression coverage for normalized pointer planning, right-click fusion flows, gaze-only click-mode execution planning, drag-origin timeout recovery, bounded drag planning, and continuous cursor-follow planning.

## 2026-03-22
- [fix] Suppressed native OpenCV camera-backend stderr noise during webcam open failures so `gaze-live` and `fusion-live` now return clean structured JSON on denied camera access.
- [test] Added regression coverage for native camera-backend stderr suppression in `tests/test_opencv_provider.py`.

## 2026-03-21
- [docs] Closed the Phase 8 review gate with a recorded review in `docs/reviews/2026-03-21-phase-8-live-multimodal-demo-review.md`.
- [feat] Added the Phase 8 live multimodal demo path: live gaze-context capture, a `fusion-live` CLI command, and a repo-local Phase 8 smoke wrapper.
- [test] Added Phase 8 tests covering missing calibration, camera failure, missing gaze lock, speech failure, and the confirmation-required live fusion flow.
- [docs] Updated MVP readiness and README guidance for the new sequential live multimodal tester path.
- [docs] Closed the Phase 7 review gate with a recorded review in `docs/reviews/2026-03-21-phase-7-webcam-mvp-completion-review.md`.
- [feat] Added the Phase 7 webcam MVP path: persisted `gaze-calibrate` and `gaze-live` commands, webcam calibration/profile storage, live webcam session helpers, and repo-local webcam wrappers.
- [test] Added Phase 7 tests covering webcam sample averaging, calibration success/failure, live gaze success, and camera failure handling.
- [docs] Closed the Phase 6 review gate with a recorded review in `docs/reviews/2026-03-21-phase-6-mvp-readiness-review.md`.
- [feat] Added the Phase 6 live macOS speech path: an Objective-C speech helper compiled on demand, a Python provider wrapper, a `voice-live` CLI command, and a repo-local Phase 6 live-voice smoke wrapper.
- [test] Added Phase 6 tests covering live speech helper build/parsing, structured permission failure handling, custom transcript confidence, and live CLI success/error paths.
- [docs] Added `docs/mvp-readiness.md` with the first tester-facing acceptance targets, setup checklist, and recommended MVP trial commands.
- [feat] Added the Phase 5 productization layer: workspace-local persistence, settings and calibration storage, structured session logging and replay, terminal overlay snapshots, a local CLI entry point, and a repo-local Phase 5 smoke wrapper.
- [test] Added Phase 5 tests covering persistence, overlay state transitions, session replay, platform capability reporting, and CLI smoke/replay flows.
- [docs] Closed the Phase 5 review gate with a recorded review in `docs/reviews/2026-03-21-phase-5-productization-review.md`.
- [feat] Added the Phase 4 multimodal fusion loop: shared interaction state, deictic target grounding, fused-confidence scoring, low-confidence clarification, execution metrics, and a repeatable fusion smoke harness.
- [test] Added Phase 4 tests covering stale-target expiry, confirmation flow, dry-run fused execution, and low-confidence clarification.
- [docs] Closed the Phase 4 review gate with a recorded review in `docs/reviews/2026-03-21-phase-4-multimodal-fusion-review.md`.
- [docs] Closed the Phase 4 spec review: fusion now targets a short-lived recent-gaze window, explicit shared state, deictic command resolution, and conservative clarification when gaze context is stale or weak.
- [feat] Added an OpenCV webcam gaze provider patch with Haar face/eye detection, coarse pupil-based gaze estimation, runtime integration, unit tests, and a live webcam smoke harness.
- [docs] Closed the webcam-provider patch review gate with a recorded review in `docs/reviews/2026-03-21-webcam-gaze-provider-patch-review.md`.
- [docs] Reopened Phase 3 with a bounded realism patch plan for a webcam-backed OpenCV gaze provider, while keeping the earlier scripted Phase 3 scope explicitly complete.
- [feat] Added the Phase 3 gaze loop: calibration fitting, smoothing, large-target inference, conservative dwell-triggered highlighting, broker integration, and a scripted gaze smoke harness.
- [docs] Closed the Phase 3 review gate with a recorded review in `docs/reviews/2026-03-21-phase-3-gaze-mvp-review.md`.
- [docs] Closed the Phase 3 spec review: seated webcam use, large targets, dwell-first triggering, and a provider-agnostic scripted gaze harness are now the reviewed scope.
- [feat] Added the Phase 2 push-to-talk voice loop: transcript streaming, bounded voice parsing, confirmation/cancellation handling, broker integration, and dry-run macOS execution flow.
- [feat] Added `scripts/phase2_voice_smoke.py` plus new audio, intent, feedback, and runtime modules for manual and automated voice-loop testing.
- [docs] Closed the Phase 2 review gate with a recorded review in `docs/reviews/2026-03-21-phase-2-voice-mvp-review.md`.
- [docs] Closed the Phase 2 spec review: push-to-talk is now the default voice interaction model, explicit command-style dictation is in scope, and target-dependent commands without gaze grounding should clarify instead of guessing.
- [feat] Finished the Phase 1 macOS control slice with placeholder `highlight_target`, point-based `focus_target`, coordinate-based confirmed click/double-click planning, and a Quartz-backed runtime helper for live point actions.
- [feat] Added `scripts/phase1_macos_smoke.py` for safe manual Phase 1 verification from the repo root.
- [docs] Closed the Phase 1 review gate with a recorded review in `docs/reviews/2026-03-21-phase-1-command-broker-review.md`.
- [docs] Locked the first bounded command vocabulary and default broker decisions in `docs/command-vocabulary.md`.
- [feat] Added a Phase 1 broker core (`interaction.control`) plus a thin dry-run macOS adapter scaffold (`interaction.platform`) for open-app, switch-app, safe keypress, and coarse scroll planning.
- [test] Added broker and macOS adapter tests covering policy defaults, confirmation flow, execution-request building, and dry-run command planning.
- [docs] Added a full phase plan hierarchy for Phases 1 through 5, including top-level spec plans, detailed subplans, and explicit code-review gate subplans.
- [docs] Encoded the plan-first, review-plans, implement, and phase-end review-gate workflow into the tracking rules and `AGENT.md`.
- [feat] Added the first typed runtime contract package in `src/interaction/contracts/` for multimodal context, intents, action proposals, broker decisions, and execution results.
- [test] Added contract-model tests covering round-trip validation and core broker policy constraints.
- [docs] Updated the roadmap and architecture docs to treat runtime contracts as the first concrete implementation slice.
- [docs] Expanded the repository from a bare scaffold into a documentation-first product foundation with product spec, interaction model, safety model, and evaluation plan.
- [docs] Locked initial direction to general computer user, macOS-first, webcam-only eye tracking, and a basic-controls showcase.
- [docs] Sharpened the roadmap, architecture, LLM-facing contract notes, and todo list around runtime contracts, command broker policy, and macOS control slices.
- [infra] Initialized the repository scaffold with `pyproject.toml`, `src/interaction/`, `tests/`, and project tracking directories.
- [docs] Mirrored the sibling project's tracking workflow in `.cursor/rules/project-tracking.mdc` and `AGENT.md`.
- [docs] Added the initial roadmap, architecture notes, todo list, bugs log, changelog, and LLM interaction guide for the voice-and-gaze computer-control concept.
- [test] Added a minimal smoke test so the scaffold has a real verification path.
