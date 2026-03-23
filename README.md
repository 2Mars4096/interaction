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

The current implementation is still intentionally limited:

- broker policy and execution-request preparation are implemented
- the macOS adapter supports a small bounded action set with dry-run as the default safety mode
- live pointer actions can now use explicit screen coordinates or normalized gaze-grounded points, and still require macOS accessibility permissions
- highlight behavior is still a placeholder notification, not a real overlay
- a push-to-talk voice loop now exists with transcript streaming, bounded voice parsing, confirmation/cancellation handling, and dry-run broker execution
- a real macOS microphone-to-transcript path now exists through a native helper exposed as `interaction voice-live`
- a real webcam calibration and live gaze path now exists through `interaction gaze-calibrate` and `interaction gaze-live`
- `gaze-live` now supports continuous `cursor` follow plus explicit dwell-triggered `highlight`, `move`, `click`, `right-click`, `double-click`, and two-stage `drag` modes for self-testing gaze-only manipulation
- a real sequential live multimodal path now exists through `interaction fusion-live`, including gaze-grounded `focus this`, `click this`, `right click this`, and `open this`
- a gaze loop now exists with calibration fitting, smoothing, large-target inference, conservative dwell-triggered highlighting, and dry-run broker execution
- an OpenCV webcam provider path now exists for coarse large-target grounding
- a multimodal fusion loop now exists with recent-gaze grounding, confidence-based clarification, confirmation handling, and repeatable dry-run metrics
- a productization layer now exists with a local CLI, session logging and replay, persisted settings/calibration, and terminal overlay snapshots
- live webcam use depends on macOS camera permission and remains heuristic rather than precision-grade
- live webcam gaze now expects a saved `webcam-live` calibration profile before the first real run
- the current fusion strategy only reasons over recent grounded targets, not a richer desktop-semantic target graph
- the current live multimodal path is sequential gaze-then-voice, not a continuous concurrent session manager
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

Installed CLI after `pip install -e ".[dev]"`:

```bash
interaction fusion-smoke
interaction gaze-smoke
interaction gaze-calibrate --settle-ms 800 --frames-per-step 6
interaction gaze-live --frames 18
interaction gaze-live --frames 24 --action cursor --execute
interaction gaze-live --frames 18 --action click --execute
interaction gaze-live --frames 24 --action drag --execute
interaction voice-live --duration 4.0
interaction fusion-live --gaze-frames 12 --duration 4.0 --confirm-duration 2.5
interaction replay --session-log .interaction/logs/fusion-smoke.jsonl
```

To execute approved live actions instead of only printing dry-run plans, add `--execute` to the relevant command:

```bash
interaction gaze-live --frames 24 --action cursor --execute
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
interaction gaze-live --frames 24 --action cursor --execute
interaction gaze-live --frames 18 --action move --execute
interaction gaze-live --frames 18 --action click --execute
interaction gaze-live --frames 24 --action drag --execute
interaction fusion-live --gaze-frames 12 --duration 4.0 --confirm-duration 2.5 --execute
```

During `gaze-live`, use `--action cursor` for continuous cursor follow, then `click`, `right-click`, `double-click`, or `drag` to test gaze-only manipulation on large targets. In `drag` mode, dwell once on the source and dwell again on the destination. During `fusion-live`, look at a large on-screen target and say commands such as `focus this`, `click this`, `right click this`, or `open this`. Confirm pointer actions when prompted.

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

At this point, `pytest` verifies the end-to-end prototype slices through the completed Phase 8 live-multimodal patch. Remaining gaps are tracked in [docs/bugs.md](docs/bugs.md) and [docs/todo.md](docs/todo.md).
