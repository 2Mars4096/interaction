# MVP Readiness

## Goal

Define when the first tester-facing macOS prototype is ready for a real user trial.

## Current Tester Path

- `interaction voice-live` or `python scripts/phase6_live_voice_smoke.py` for real microphone capture and bounded voice commands; add `--execute` to run approved live actions
- `interaction gaze-calibrate` or `python scripts/phase7_webcam_calibrate.py` for live webcam calibration
- `interaction gaze-live` or `python scripts/phase7_live_webcam_smoke.py` for the live webcam gaze path; use `--action cursor|move|click|right-click|double-click|drag` for explicit gaze-only manipulation tests
- `interaction fusion-live` or `python scripts/phase8_live_fusion_smoke.py` for live commands with concurrent webcam sensing during speech and confirmation; add `--turns N` or `--continuous` to keep one session open and `--execute` to run approved live pointer actions
- `interaction gaze-smoke` or `python scripts/phase5_productization_smoke.py gaze-smoke` for the scripted gaze baseline
- `python scripts/phase3_webcam_smoke.py --frames 12` for the lower-level webcam provider diagnostic
- `interaction fusion-smoke` for the repeatable dry-run fusion path
- `interaction voice-live --continuous --speech-mode stream --duration 8` and the equivalent `fusion-live` command for opt-in real transcript previews and silence endpointing

## Acceptance Targets

- **Voice latency:** the live voice path should either return a final transcript or a structured failure message within 10 seconds of starting a 4-second capture window.
- **Streaming trial:** measure time from end of speech to valid final feedback separately from microphone startup. Verify that natural pauses are not clipped, silence completes short turns before the maximum, and an active turn reaching the maximum never executes. Compare against recorded mode; automated checks do not establish this latency or accuracy.
- **Webcam calibration:** save only after complete timed samples at all five visible fitting points and four separate validation points satisfy error and mapped-dispersion gates; failures/cancellation return structured feedback and preserve the previous profile.
- **Webcam gaze startup:** the live webcam path should either begin producing target/recovery events or fail structurally within 5 seconds of startup.
- **Large-target gaze reaction:** the scripted and webcam-backed gaze paths should highlight a stable large target within about 1 second of dwell when permissions and lighting are acceptable.
- **Accidental actions:** `L2` and `L3` actions must never auto-execute without confirmation in the MVP test flow.
- **Recovery quality:** missing microphone, speech, camera, or accessibility permission should yield a structured recovery message rather than a traceback.
- **Voice-command accuracy target:** on a 10-command manual checklist of bounded commands, the first tester path should succeed or clarify on at least 8 of 10 runs.
- **Wrong-target tolerance:** deictic commands should clarify rather than execute when gaze context is stale or confidence is below the current threshold.
- **Live multimodal recovery:** `fusion-live` should either ground gaze plus capture speech or fail structurally within one run without a traceback.

## Tester Checklist

1. Grant microphone permission when prompted by the live speech helper.
2. Grant speech-recognition permission when prompted by the live speech helper.
3. Grant camera permission if testing the webcam gaze provider.
4. Run `interaction gaze-calibrate` before `interaction gaze-live`.
5. Run `interaction gaze-calibrate` before `interaction fusion-live`.
6. Grant accessibility permission if testing live pointer actions outside dry-run mode.
7. Leave `dry_run` enabled in `.interaction/settings.json` for the first test pass, then use `--execute` on a per-run basis once permissions are granted.

## Recommended First Trial

```bash
python scripts/phase6_live_voice_smoke.py --duration 4.0
python scripts/phase7_webcam_calibrate.py --settle-ms 800 --frames-per-step 6
python scripts/phase7_live_webcam_smoke.py --frames 24 --action cursor --execute
python scripts/phase7_live_webcam_smoke.py --frames 18 --action click --execute
python scripts/phase7_live_webcam_smoke.py --frames 24 --action drag --execute
python scripts/phase8_live_fusion_smoke.py --gaze-frames 12 --duration 4.0 --confirm-duration 2.5 --execute
interaction fusion-smoke
```

## Notes

- The tester-facing path includes bounded/continuous voice and fusion command sessions, live webcam gaze, cursor follow, and explicit dwell manipulation. Fusion keeps one camera worker active across commands and confirmations. Webcam gaze remains heuristic and region-based; speech supports recorded windows and opt-in streaming, and native hardware performance remains unmeasured by the automated checks.
- The cursor-follow path exposes tunable `deadzone`, `smoothing`, `edge_padding`, and `max_step` controls for per-machine adjustment during manual testing.
- The gaze-only dwell path now also exposes `action_cooldown_ms` and `drag_timeout_ms` controls so testers can reduce repeated clicks and recover cleanly from an abandoned drag arm.
- This document closes the backlog item around acceptable latency, accuracy, and accidental-action targets for the first MVP trial.
