# Architecture

## Status

Prototype implementation exists through Phase 4. This document now captures both the intended system shape and the concrete modules already in place.

## Working Assumptions

- **Core language:** Python 3.11+
- **Primary development and target platform:** macOS
- **Cross-platform intent:** yes, but later
- **Tracking hardware for v1:** standard webcam
- **Speech stack:** undecided
- **Gaze / eye-tracking stack:** undecided
- **Desktop UI / overlay surface:** terminal overlay model implemented; native macOS overlay still deferred

Do not lock in vendor or library choices until the contracts phase resolves latency targets, calibration strategy, and macOS execution boundaries.

## Intended System Shape

### 1. Input Layer

- Audio stream capture
- Webcam or dedicated eye-tracker stream capture
- Optional keyboard / mouse fallback events

### 2. Perception Layer

- Speech-to-text / streaming transcription
- Voice activity detection and turn segmentation
- Gaze estimation, smoothing, and calibration
- Eye gesture or facial micro-gesture detection

### 3. Intent Layer

- Multimodal state assembly
- Intent parsing and normalization
- Target grounding from gaze context
- Confidence estimation and ambiguity detection

### 4. Command Broker

- Converts normalized intents into bounded action proposals
- Applies safety policy, confirmation thresholds, and platform constraints
- Produces auditable action requests and action results

### 5. Platform Adapter Layer

- macOS window and application actions
- Cursor, scroll, click, and keypress primitives
- App-specific adapters where needed

### 6. Feedback Layer

- Spoken acknowledgement
- Visual overlays for current target, confidence, and pending confirmation
- Logs for what the system heard, inferred, and executed

## Runtime Contracts To Establish Early

- **Observation events:** transcript deltas, final transcript segments, gaze frames, calibration state, active-app context
- **Intent candidates:** normalized user goal plus target references and confidence
- **Action proposals:** bounded verb, typed arguments, safety classification, and rationale
- **Broker decisions:** allow, confirm, clarify, reject
- **Execution results:** success, failure, cancelled, blocked

The implementation should keep these contracts explicit and typed before integrating any heavy model or CV dependency.

The first concrete code in the repository should live here, inside `src/interaction/contracts/`, so later perception, broker, and platform modules all share the same schema surface.

## Repository Layout

```text
docs/              Planning, roadmap, and tracking memory
docs/plans/        Active and completed implementation plans
src/interaction/   Python package for runtime code and contracts
tests/             Automated tests
```

Expected subpackages once implementation begins:

```text
src/interaction/
  contracts/       Shared schemas for events, intents, proposals, results
  audio/           Speech capture and transcription
  vision/          Gaze estimation, calibration, eye gestures
  intent/          Multimodal interpretation and normalization
  control/         Command broker, safety policy, action contracts
  platform/        OS and application adapters
  feedback/        Spoken and on-screen responses
  runtime/         Session state, event bus, orchestration loop
```

## Conventions

- Keep perception, interpretation, and execution separate.
- Never let raw LLM output directly trigger medium- or high-risk desktop actions.
- Prefer typed action proposals and typed action results over free-form strings.
- Log both raw observations and normalized decisions so behavior is debuggable.
- Design first for large, forgiving targets because webcam gaze precision will be limited.
- Keep the platform layer thin and auditable; broker policy should not be buried inside OS adapters.

## Current Implementation Notes

- `src/interaction/contracts/` provides the shared typed schema layer.
- `src/interaction/control/` now contains the first broker policy engine and execution-request preparation logic.
- `src/interaction/platform/macos.py` now supports bounded plans for `open_app`, `switch_app`, safe `press_key`, coarse `scroll`, notification-based `highlight_target`, point-based `focus_target`, and coordinate-based or normalized-gaze-based `click_target` / `double_click_target` / `right_click_target` / `drag_target`.
- `src/interaction/platform/macos_runtime.py` contains the minimal Quartz-based cursor move/click/right-click/drag helper used for live point-based actions.
- `scripts/phase1_macos_smoke.py` is the safe manual smoke harness for this phase.
- `src/interaction/audio/` now contains the Phase 2 transcript-stream and push-to-talk turn primitives.
- `src/interaction/audio/macos_speech.py` now provides the Phase 6 macOS live speech wrapper, and `src/interaction/audio/macos_speech_bridge.m` is the small native helper source compiled on demand into `.interaction/bin/`.
- `src/interaction/intent/voice.py` provides the bounded pattern-based voice interpreter.
- `src/interaction/feedback/voice.py` provides event-level voice feedback states.
- `src/interaction/runtime/voice.py` coordinates transcript segments, interpretation, broker decisions, confirmations, cancellations, and execution.
- `scripts/phase2_voice_smoke.py` is the safe manual smoke harness for the voice loop.
- `src/interaction/vision/` now contains the Phase 3 normalized calibration, smoothing, target-inference, and dwell-trigger primitives.
- `src/interaction/vision/opencv_provider.py` adds a coarse OpenCV webcam provider using Haar face/eye detection plus pupil heuristics.
- `src/interaction/feedback/gaze.py` provides event-level gaze feedback states.
- `src/interaction/runtime/gaze.py` coordinates calibration, smoothing, target inference, conservative dwell triggers, and broker/macOS dry-run execution.
- `src/interaction/runtime/webcam.py` now coordinates live webcam calibration targets, persisted calibration fitting, default live target regions, and the live webcam gaze session helper.
- `scripts/phase3_gaze_smoke.py` is the safe manual smoke harness for the gaze loop.
- `scripts/phase3_webcam_smoke.py` is the live webcam smoke harness for the OpenCV provider patch.
- `src/interaction/runtime/state.py` now holds the shared multimodal interaction state for recent transcripts, gaze context, candidate targets, and freshness windows.
- `src/interaction/intent/fusion.py` resolves deictic utterances such as `"open this"` and `"click that"` against recent grounded gaze targets.
- `src/interaction/feedback/fusion.py` provides event-level fusion feedback states.
- `src/interaction/runtime/fusion.py` coordinates shared-state fusion, confidence aggregation, clarification, confirmation, and dry-run execution metrics.
- `scripts/phase4_fusion_smoke.py` is the repeatable dry-run smoke harness for the fusion loop.
- `src/interaction/ui/overlay.py` now provides the structured overlay-state model plus a terminal renderer for runtime visibility.
- `src/interaction/session/logging.py` now provides structured JSONL session logging and replay helpers.
- `src/interaction/persistence/store.py` now provides workspace-local runtime paths, persisted settings, and calibration-profile storage.
- `src/interaction/app.py` now provides the local CLI entry point for repeatable voice, gaze, fusion, and replay flows.
- `scripts/phase5_productization_smoke.py` is the repo-local smoke wrapper for the Phase 5 CLI.
- `scripts/phase6_live_voice_smoke.py` is the repo-local smoke wrapper for the live macOS voice path.
- `scripts/phase7_webcam_calibrate.py` and `scripts/phase7_live_webcam_smoke.py` are the repo-local wrappers for the webcam calibration and live-gaze tester paths.
