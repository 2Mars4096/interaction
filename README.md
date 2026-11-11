# Interaction

Multimodal computer control centered on two input channels:

- realtime voice interactions that can translate natural language into safe computer actions
- gaze and eye behavior for pointing, target selection, focus, and lightweight control

The current product direction is now sharper:

- **Target user:** general computer user
- **Platform:** macOS first, cross-platform later
- **Hardware:** normal webcam only for the first prototype
- **Showcase:** basic controls that can start replacing parts of keyboard and mouse use

## Status

This repository now contains a working prototype stack through multimodal fusion: typed contracts, broker policy, a bounded macOS adapter, a push-to-talk voice loop, a gaze loop, a webcam-backed coarse gaze provider, and a shared-state fusion loop for commands like `"open this"` and `"click this"`.

The Phase 9 reliability pass adds repeated live voice sessions, fresh concurrent gaze, validated calibration, tracking-loss recovery, and reproducible packages. Its 476-test suite and eight Linux/macOS CI jobs passed, including native speech-helper compilation from the installed wheel. Hardware accuracy and live interaction evaluation remain open; see the [completion plan](docs/plans/12-reliable-live-interaction.md).

Phase 10 is complete under its documented scope: continuous fusion sessions, bounded retained history with complete event journals, and opt-in streaming speech with real previews and silence endpointing. Its 623-test suite and all eight Linux/macOS [CI jobs passed](https://github.com/2Mars4096/interaction/actions/runs/42994103807), including installed-wheel native compilation and argument probes. See the [responsive sessions plan](docs/plans/13-responsive-live-sessions.md) for section-level verification and remaining hardware checks.

The current implementation is still intentionally limited:

- broker policy and execution-request preparation are implemented
- the macOS adapter supports a small bounded action set with dry-run as the default safety mode
- live pointer actions can now use explicit screen coordinates or normalized gaze-grounded points, and still require macOS accessibility permissions
- highlight behavior is still a placeholder notification, not a real overlay
- the voice loop supports bounded command parsing, confirmation/cancellation handling, and dry-run broker execution; live sessions support recorded or opt-in streaming capture
- a real macOS microphone-to-transcript path now exists through a native helper exposed as `interaction voice-live`
- a real webcam calibration and live gaze path now exists through `interaction gaze-calibrate` and `interaction gaze-live`
- `gaze-live` now supports continuous `cursor` follow plus explicit dwell-triggered `highlight`, `move`, `click`, `right-click`, `double-click`, and two-stage `drag` modes for self-testing gaze-only manipulation
- `interaction fusion-live` keeps webcam gaze capture active during speech and confirmation, including gaze-grounded `focus this`, `click this`, `right click this`, and `open this`
- a gaze loop now exists with calibration fitting, smoothing, large-target inference, conservative dwell-triggered highlighting, and dry-run broker execution
- an OpenCV webcam provider path now exists for coarse large-target grounding
- a multimodal fusion loop now exists with recent-gaze grounding, confidence-based clarification, confirmation handling, and repeatable dry-run metrics
- a productization layer now exists with a local CLI, session logging and replay, persisted settings/calibration, and terminal overlay snapshots
- live webcam use depends on macOS camera permission and remains heuristic rather than precision-grade
- live webcam gaze requires a current validated `webcam-live` profile for the selected camera
- the current fusion strategy only reasons over recent grounded targets, not a richer desktop-semantic target graph
- live fusion supports single, bounded, or continuous command sessions while a single camera worker maintains fresh gaze context
- the overlay is terminal-rendered for now, not a native always-on-top macOS overlay

Manual Phase 1 smoke harness:

```bash
python scripts/phase1_macos_smoke.py --action open_app --app-name Safari
python scripts/phase1_macos_smoke.py --action click_target --risk L2 --requires-confirmation --confirm --target-ref target_1 --x 500 --y 300
```

Manual Phase 2 voice smoke harness:

```bash
python scripts/phase2_voice_smoke.py --utterance "open Safari"
python scripts/phase2_voice_smoke.py --utterance "type hello world" --follow-up yes
python scripts/phase2_voice_smoke.py --utterance "translate hello to French"
```

Manual Phase 3 gaze smoke harness:

```bash
python scripts/phase3_gaze_smoke.py
python scripts/phase3_webcam_smoke.py --frames 12
```

Manual Phase 4 fusion smoke harness:

```bash
python scripts/phase4_fusion_smoke.py
```

Manual Phase 5 productization smokes:

```bash
python scripts/phase5_productization_smoke.py fusion-smoke
python scripts/phase5_productization_smoke.py gaze-smoke
python scripts/phase5_productization_smoke.py replay --session-log .interaction/logs/fusion-smoke.jsonl
```

Manual Phase 6 live-voice smoke:

```bash
python scripts/phase6_live_voice_smoke.py --duration 4.0
```

Manual Phase 7 webcam MVP commands:

```bash
python scripts/phase7_webcam_calibrate.py --settle-ms 800 --frames-per-step 6
python scripts/phase7_live_webcam_smoke.py --frames 18
```

Manual Phase 8 live multimodal demo:

```bash
python scripts/phase8_live_fusion_smoke.py --gaze-frames 12 --duration 4.0 --confirm-duration 2.5
```

Installed CLI after the development setup below:

```bash
interaction fusion-smoke
interaction gaze-smoke
interaction gaze-calibrate --settle-ms 800 --frames-per-step 6
interaction gaze-live --frames 18
interaction gaze-live --frames 24 --action cursor --execute --show-dot
interaction gaze-live --frames 18 --action click --execute
interaction gaze-live --frames 24 --action drag --execute
interaction voice-live --duration 4.0
interaction fusion-live --gaze-frames 12 --duration 4.0 --confirm-duration 2.5
interaction replay --session-log .interaction/logs/fusion-smoke.jsonl
```

To execute approved live actions instead of only printing dry-run plans, add `--execute` to the relevant command:

```bash
interaction gaze-live --frames 24 --action cursor --execute --show-dot
interaction gaze-live --frames 18 --action click --execute
interaction gaze-live --frames 24 --action drag --execute
interaction voice-live --duration 4.0 --execute
interaction fusion-live --gaze-frames 12 --duration 4.0 --confirm-duration 2.5 --execute
```

Recommended working demo on macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
interaction gaze-calibrate --settle-ms 800 --frames-per-step 6
interaction gaze-live --frames 24 --action cursor --execute --show-dot
interaction gaze-live --frames 18 --action move --execute
interaction gaze-live --frames 18 --action click --execute
interaction gaze-live --frames 24 --action drag --execute
interaction fusion-live --gaze-frames 12 --duration 4.0 --confirm-duration 2.5 --execute
```

During `gaze-live`, use `--action cursor` for continuous cursor follow, then `click`, `right-click`, `double-click`, or `drag` to test gaze-only manipulation on large targets. Add `--show-dot` to render a small live gaze marker on screen during the run. In `drag` mode, dwell once on the source and dwell again on the destination. During `fusion-live`, look at a large on-screen target and say commands such as `focus this`, `click this`, `right click this`, or `open this`. Confirm pointer actions when prompted.

If `cursor` mode feels too twitchy or too sluggish, tune it per run:

```bash
interaction gaze-live --frames 24 --action cursor --execute \
  --cursor-deadzone 0.02 \
  --cursor-smoothing 0.24 \
  --cursor-edge-padding 0.04 \
  --cursor-max-step 0.14
```

If dwell actions feel too eager, raise the cooldown or extend the drag-arm timeout:

```bash
interaction gaze-live --frames 18 --action click --execute --action-cooldown-ms 1200
interaction gaze-live --frames 24 --action drag --execute --action-cooldown-ms 1200 --drag-timeout-ms 4000
```

## Working Product Shape

### Live voice sessions

Start one session for successive commands:

```bash
interaction voice-live --continuous --execute
interaction voice-live --turns 3
```

Say `stop listening` or press Ctrl-C to finish. Required confirmations are captured in the same session, with a visible prompt before listening; `cancel` abandons the action. Live feedback is written to stderr immediately, session events are saved as they occur, and stdout remains the final JSON result. A one-command run remains the default.

Commands accept ordinary sentence punctuation and polite prefixes. `click`, `right click`, and `double click` refer to gaze targets in fusion; voice-only use asks for a target. Dictated punctuation and spacing are preserved. Explicitly low-confidence recognition asks for a repeat instead of executing.

Recorded capture remains the default. Continuous sessions repeat the selected capture mode; a native desktop shell remains separate work.

### Streaming speech previews and turn completion

Enable the streaming path to see real transcript updates while speaking:

```bash
interaction voice-live --continuous --speech-mode stream --duration 8 --silence-duration 0.8
interaction fusion-live --continuous --speech-mode stream --duration 8 --silence-duration 0.8
```

Wait for `Microphone ready`, then speak. `Hearing:` lines are revisable previews. Only a valid final result from a successfully exited helper reaches the command loop. The same mode is used for confirmations, and `stop listening` or Ctrl-C ends the session. The commands above keep the existing dry-run behavior; add `--execute` when testing actual desktop actions.

In streaming mode, `--duration` sets the maximum for each primary command (default 4 seconds), and `--confirm-duration` sets a separate maximum for confirmations (default 2.5 seconds). The `--duration 8` examples above leave that confirmation maximum unchanged. Capture can end after the configured silence interval (0.2–5 seconds, default 0.8) or an actual recognizer final. If the ceiling cuts off active speech, the command is discarded and continuous mode asks for a repeat; increasing the maximum does not impose that whole wait on short commands. Recognition finalization has a separate eight-second limit. A partial, timeout, malformed event, or failed helper exit cannot authorize an action.

Capture metadata reports the measured duration, configured maximum, and endpoint reason. Silence uses an initial audio-energy heuristic; microphone sensitivity, noise, recognition quality, and end-to-end latency still require testing on your Mac. Streaming is opt-in for that comparison. Use `--speech-mode recorded` for the existing fixed-window path. Each turn owns its microphone capture; this is not a wake-word or always-open audio service.

### Repeated voice-and-gaze commands

After calibration, keep one camera session open for several commands:

```bash
interaction fusion-live --turns 3
interaction fusion-live --continuous
interaction fusion-live --continuous --confirmation-attempts 2 --execute
```

Say `stop listening` or press Ctrl-C to finish; `cancel` abandons only the pending action. Continuous sessions recover from silence and foreground changes so you can repeat a command. A confirmation has a bounded retry budget, and exhausting it cancels the pending action. Permission and camera failures end the session with structured feedback. The default remains one command.

Each actionable command and follow-up uses current gaze and foreground context. Loss of tracking after startup allows voice-only commands and session stop, while pointer commands must regain a valid target. Final JSON includes command/confirmation captures and `session` counts, with the existing last-capture fields retained. `session.commands` counts successful primary captures, including a spoken stop; fusion's separate `metrics.executions` counts executed action results.

### Session history and replay

Live voice, fusion, gaze, and calibration sessions retain the latest 200 events by default. Voice and fusion also retain the latest 200 captures of each kind. Use `--history-limit N` to change the positive retention limit. Final JSON's `history` reports total, retained, and dropped counts; session and execution counts still cover the whole session. This limit changes retained output, not active pending actions or gaze freshness.

The JSONL event journal remains complete and is written as events occur. Use `interaction replay --session-log <path>` to inspect the complete event history; explicit replay loads that journal in full. Capture arrays in the final JSON are recent summaries, while the journal contains the full sequence of feedback, transcripts, decisions, and results.

### Live context and cancellation

For actionable speech, voice and fusion sessions inspect the foreground application and window before and after capture. If that context changes, repeat the command in the intended location. Live execution requires an identified foreground application; the probe may need macOS Automation/Accessibility access. A recognized `stop listening` cancels pending work before post-capture context or gaze validation, so switching windows or losing tracking cannot prevent session stop.

`fusion-live` keeps a single camera worker running during speech. It selects the recent stable gaze target when the command finishes. Confirmations expire after 15 seconds and cancel on a changed application/window, lost or stale gaze, a different target, or excessive movement from the pending point. Returning to the old target does not revive a cancelled action. `--gaze-delta-ms` controls the sampling delay; timestamps and freshness use actual elapsed time.

These checks identify an application/window and coarse screen region. Exact controls and editable fields still require Accessibility target integration.

### Tracking loss and dwell

Live capture uses measured monotonic time; `--delta-ms` is a minimum sampling delay. A camera stall does not count as time looking at a target. Missing, weak, invalid, or out-of-screen readings clear partial dwell and cancel an armed drag. Live gaze feedback is delivered immediately, and Ctrl-C ends the session with cleanup.

A dwell action fires once while you keep looking at the same target. Waiting out the cooldown or briefly losing tracking does not rearm it. Look away with valid tracking for at least 200 ms, then complete a fresh dwell at the desired target. Valid look-away can be another coarse region or an area outside the current target set.

Fusion and cursor follow require at least three samples over 120 ms, within a 0.05 normalized-point radius, with no sampling gap above 250 ms. These checks use unsmoothed positions. Returned lock evidence reports sample count and duration; the confidence value remains a detector heuristic, separate from calibration error measurements.

### Visible calibration and validation

Run `interaction gaze-calibrate` on the primary display. A visible dot moves through five fitting positions, then four different positions to check the fitted mapping. Keep your head comfortably still and look at each dot; press Escape or Ctrl-C to cancel. Tk and a desktop display are required. Progress appears immediately in the terminal as well as on screen.

Each position needs the full requested sample set (at least three frames). Calibration supports reversed axes and rejects weak signal ranges, incomplete samples, excessive mapped jitter, or poor results on the independent points. The initial acceptance limits are a mean normalized error of 0.12, a worst-point error of 0.20, and a maximum per-axis mapped standard deviation of 0.05. These are coarse-target engineering checks, not a claim of precise cursor accuracy.

Only a successful validation replaces the saved profile. Failed or cancelled attempts preserve the previous file. Live modes reject older profiles without validation evidence and profiles from a different camera index or tracker schema; rerun calibration after updating. Recalibrate when camera position, display setup, or seating changes. Calibration and pointer conversion currently address the primary display only.

- **Voice for intent**: "open that", "translate this", "scroll slower", "reply in Chinese"
- **Gaze for targeting**: "that" is grounded by what the user is looking at
- **Command broker for safety**: the system proposes and validates actions before sending them to the OS
- **Feedback loop for trust**: visual and spoken acknowledgements explain what the system heard, inferred, and did

## Documentation Set

- [docs/product-spec.md](docs/product-spec.md) - product goals, scope, user, and MVP definition
- [docs/interaction-model.md](docs/interaction-model.md) - how voice, gaze, confirmation, and feedback fit together
- [docs/safety-model.md](docs/safety-model.md) - action risk taxonomy and broker policy
- [docs/command-vocabulary.md](docs/command-vocabulary.md) - first bounded command set and default broker decisions
- [docs/evaluation-plan.md](docs/evaluation-plan.md) - success metrics for latency, accuracy, accidental actions, and usability
- [docs/mvp-readiness.md](docs/mvp-readiness.md) - tester-facing acceptance targets, setup checklist, and first-trial commands
- [docs/speech-stream-protocol.md](docs/speech-stream-protocol.md) - bounded native event protocol, endpointing, and final-only execution
- [docs/development-plan.md](docs/development-plan.md) - roadmap from contracts to productization
- [docs/architecture.md](docs/architecture.md) - intended system shape and repository conventions
- [docs/reviews/README.md](docs/reviews/README.md) - phase-end code review gate convention

## Repo Layout

```text
.
├── AGENT.md
├── docs/
│   ├── architecture.md
│   ├── bugs.md
│   ├── changelog.md
│   ├── development-plan.md
│   ├── evaluation-plan.md
│   ├── interaction-model.md
│   ├── llm-api-guide.md
│   ├── plans/
│   ├── product-spec.md
│   ├── reviews/
│   ├── safety-model.md
│   └── todo.md
├── src/interaction/
│   ├── contracts/
│   ├── persistence/
│   ├── session/
│   └── ui/
└── tests/
```

## Quick Start

The CI baseline exercises Python 3.11, 3.12, and 3.13 on Linux and macOS. Use Python 3.12 for the following development setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

The compatibility ranges in `pyproject.toml` exclude OpenCV 5, which removed APIs used by this tracker. `requirements-dev.lock` records the exact runtime, test, and build dependency baseline with hashes. To refresh it deliberately, use the command in its header and review changes to `constraints-test.txt`.

Live microphone capture requires macOS and the Xcode command-line tools (`clang`). Calibration needs Tk and a desktop display. The speech helper packages and embeds its microphone/speech usage descriptions, builds into a cache identified by source/platform information, and reports build/capture failures with finite deadlines. Parent-owned temporary recordings are removed when capture finishes, fails, times out, or is interrupted.

## Build and CI

```bash
python -m ruff check --select E9,F63,F7,F82 src tests scripts
python -m build --no-isolation
python scripts/verify_distribution.py --dist-dir dist
python scripts/verify_installed_package.py --dist-dir dist --requirements requirements-dev.lock
```

The default build produces a source distribution and builds the wheel from that archive using the already locked build tools. Verification checks the native helper and permission metadata in both archives. The installed-package probe creates a separate environment outside the checkout, installs the wheel and hashed dependencies, and checks imports, the CLI entry point, and OpenCV's Haar resources.

On macOS, add `--compile-native` to the installed-package check to compile the installed helper, verify cache reuse, and probe streaming argument errors before permission requests or audio startup. GitHub Actions runs unit tests across the Python/platform matrix and the distribution checks on Linux/macOS. These checks do not open the camera, record audio, request desktop permissions, or move the pointer.

Automated checks cover the Phase 9 reliability work and Phase 10 responsive sessions; camera accuracy, streaming endpoint behavior, and native interaction still need hardware evaluation. Remaining gaps are tracked in [docs/bugs.md](docs/bugs.md) and [docs/todo.md](docs/todo.md).
