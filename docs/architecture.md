# Architecture

## Status

The implementation includes the Phase 9 reliability repairs and Phase 10 responsive live sessions: continuous fusion, bounded retained history, and opt-in streaming speech. Verification evidence is recorded in the phase plans; hardware evaluation remains open. This document captures the intended system shape and the concrete modules already in place.

## Working Assumptions

- **Core language:** Python 3.11+
- **Primary development and target platform:** macOS
- **Cross-platform intent:** yes, but later
- **Tracking hardware for v1:** standard webcam
- **Speech stack:** Apple Speech and AVFoundation through an Objective-C helper; recorded capture is the default, with opt-in AVAudioEngine streaming
- **Gaze / eye-tracking stack:** OpenCV Haar face/eye detection and pupil heuristics, with calibrated coarse targets and stability gates
- **Desktop UI / overlay surface:** terminal overlay model implemented; native macOS overlay still deferred

Keep speech and gaze providers behind the typed runtime contracts so later replacements can be compared without changing broker policy or macOS execution boundaries.

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
- Native speech resources include `macos_speech_info.plist` with microphone/speech usage descriptions embedded by the linker. Helper caching uses source, metadata, build flags, and OS/architecture identity; compilation and capture are bounded subprocesses.
- Speech recordings live in a parent-owned temporary directory, allowing cleanup when a child exits, times out, or is interrupted. Native payload types are validated before constructing transcript events.
- Optional stream mode uses AVAudioEngine buffers and SFSpeechAudioBufferRecognitionRequest directly, without a recording file. Audio energy and real quiet buffers determine a silence endpoint; an active utterance cut off by the capture maximum is rejected. Finalization has a separate bounded grace period.
- `audio/streaming.py` validates versioned JSONL events and drains stdout/stderr with line, byte, event, and lifetime caps. It returns executable text only after one valid final, complete output, and exit zero. Timeout, Ctrl-C, callback failure, and malformed output terminate/reap the helper and close pipes.
- Native stream callbacks publish display-only `SpeechStreamEvent` updates; `LiveEventBuffer` shows and journals previews immediately. Live final transcripts bypass scripted word-prefix simulation; smoke paths keep it.
- `src/interaction/intent/voice.py` provides the bounded pattern-based voice interpreter.
- `src/interaction/feedback/voice.py` provides event-level voice feedback states.
- `src/interaction/runtime/voice.py` coordinates transcript segments, interpretation, broker decisions, confirmations, cancellations, and execution.
- `scripts/phase2_voice_smoke.py` is the safe manual smoke harness for the voice loop.
- `src/interaction/vision/` now contains the Phase 3 normalized calibration, smoothing, target-inference, and dwell-trigger primitives.
- `src/interaction/vision/opencv_provider.py` adds a coarse OpenCV webcam provider using Haar face/eye detection plus pupil heuristics.
- `src/interaction/feedback/gaze.py` provides event-level gaze feedback states.
- `src/interaction/runtime/gaze.py` coordinates calibration, smoothing, target inference, conservative dwell triggers, and broker/macOS dry-run execution.
- `src/interaction/runtime/webcam.py` now coordinates live webcam calibration targets, persisted calibration fitting, default live target regions, and the live webcam gaze session helper.
- `src/interaction/vision/calibration.py` fits signed per-axis least-squares mappings and retains typed held-out error/dispersion evidence in versioned profiles. Unclamped predictions are used for validation.
- `src/interaction/ui/calibration.py` renders primary-display reference targets on the main thread, pumps events during settling/sampling, and handles Escape/close cancellation.
- Calibration uses five fitting points followed by four disjoint validation points. The webcam aggregate reports real sample duration and per-axis dispersion; mapped jitter is checked after fitting.
- Live profile loading checks schema, tracker signal, camera index, and recomputed validation evidence. Atomic profile writes preserve the prior file on unsuccessful calibration.
- Normalized macOS pointer conversion uses Quartz global display bounds, matching the calibration screen's logical primary-display coordinates instead of Retina backing pixel dimensions.
- `scripts/phase3_gaze_smoke.py` is the safe manual smoke harness for the gaze loop.
- `scripts/phase3_webcam_smoke.py` is the live webcam smoke harness for the OpenCV provider patch.
- `src/interaction/runtime/state.py` now holds the shared multimodal interaction state for recent transcripts, gaze context, candidate targets, and freshness windows.
- `src/interaction/runtime/clock.py` supplies the injected monotonic clock; worker observations retain their original timestamps. Explicit `advance()` remains a scripted simulation facility.
- `src/interaction/runtime/pending.py` binds confirmation to the proposal's creation time, environment, target revision, and gaze point. Tracking loss cancels pending pointer actions.
- `src/interaction/runtime/live_gaze.py` owns camera open/read/close on one background worker while the main thread captures speech and interprets commands. Stable snapshots carry freshness and loss/target-change revisions.
- `src/interaction/vision/stability.py` shares unsmoothed stable-target evidence across the worker, cursor follow, and bounded context capture: sample count, duration, confidence, displacement, gap, and revision.
- `src/interaction/runtime/sampling.py` paces synchronous live reads and records measured timestamps. Live gaze separates elapsed time for timeouts/cooldown from observed same-target fixation time.
- Dwell preserves its fired-target latch across tracking loss and cooldown; sustained valid look-away rearms it. Missing/weak/gapped tracking resets smoothing and partial dwell and cancels armed drag.
- Target confidence no longer adds a constant boost to detector evidence. Calibration quality, tracking continuity, and detection confidence remain distinct signals.
- `src/interaction/platform/environment.py` performs a bounded, read-only macOS foreground app/window probe; unavailable values remain unknown. Live execution requires identified app context.
- `src/interaction/intent/fusion.py` resolves deictic utterances such as `"open this"` and `"click that"` against recent grounded gaze targets.
- `src/interaction/feedback/fusion.py` provides event-level fusion feedback states.
- `src/interaction/runtime/fusion.py` coordinates shared-state fusion, confidence aggregation, clarification, confirmation, and dry-run execution metrics.
- `scripts/phase4_fusion_smoke.py` is the repeatable dry-run smoke harness for the fusion loop.
- `src/interaction/ui/overlay.py` now provides the structured overlay-state model plus a terminal renderer for runtime visibility.
- `src/interaction/session/logging.py` now provides structured JSONL session logging and replay helpers.
- `src/interaction/session/live.py` publishes live CLI feedback to stderr, journals events immediately, and updates overlay state without duplicating records when the final JSON payload is built.
- Its `BoundedHistory` retains the most recent entries with exact lifetime counts. Live events, serialized output, overlay snapshots, and speech capture summaries use bounded retention; JSONL feedback journals are complete. `--history-limit` configures live voice/fusion/gaze/calibration retention, and final `history` metadata makes omissions explicit.
- `voice-live` can retain one command loop for bounded multiple commands or `--continuous` sessions, including required confirmation turns. Both voice and fusion select recorded or opt-in streamed capture using the same helper and policy boundaries.
- `fusion-live` uses the same session options while retaining one camera worker for its entire lifetime. Startup requires calibration and an initial gaze lock; each later turn refreshes gaze and foreground context. Retryable silence/context changes do not restart the camera, and shutdown clears pending actions before joining the worker.
- `src/interaction/persistence/store.py` now provides workspace-local runtime paths, persisted settings, and calibration-profile storage.
- `src/interaction/app.py` now provides the local CLI entry point for repeatable voice, gaze, fusion, and replay flows.
- `scripts/phase5_productization_smoke.py` is the repo-local smoke wrapper for the Phase 5 CLI.
- `scripts/phase6_live_voice_smoke.py` is the repo-local smoke wrapper for the live macOS voice path.
- `scripts/phase7_webcam_calibrate.py` and `scripts/phase7_live_webcam_smoke.py` are the repo-local wrappers for the webcam calibration and live-gaze tester paths.
- `requirements-dev.lock` is the hashed universal dependency baseline; `constraints-test.txt` records deliberately selected runtime/test tool versions. Project compatibility ranges stay in `pyproject.toml`.
- `scripts/verify_distribution.py` checks packaged native resources in wheel/sdist. `scripts/verify_installed_package.py` creates a clean environment outside the checkout and verifies imports, entry point, and native resources. Optional macOS checks compile the installed helper, verify cache reuse, and probe streaming argument errors before permission requests or audio startup.
- GitHub Actions covers Linux/macOS Python 3.11–3.13 unit tests and separate distribution/native-compilation checks. CI uses locked build tools without build isolation and does not invoke hardware input or desktop actions.
