# Known Issues & Failed Approaches

## Open Bugs

- Phase 9 fixes are covered by automated checks. Remaining product and hardware limitations are listed below.

## Resolved in Phase 10

- Section 2: live events, snapshots, and speech capture summaries no longer grow for the full lifetime of continuous sessions; configurable retention and explicit truncation counts preserve complete event replay.
- Section 1: live fusion no longer ends after one command when `--turns` or `--continuous` is requested; confirmation silence has a bounded retry budget, and camera ownership spans the full session.
- Section 1: successful reacquisition removes stale tracking-loss error fields from the final gaze status.
- Section 1: spoken session stop is handled independently of foreground/target action authorization, so focus changes cannot discard a stop request.

## Resolved in Phase 9

- Section 5: OpenCV compatibility is bounded below 5, the tested dependency baseline is hashed, and wheel/sdist/install checks require the native helper and permission metadata.
- Section 5: native helper deadlines, strict payload validation, atomic versioned caching, and parent-owned audio cleanup cover build/runtime failure paths.
- Section 4: dropped/weak tracking and long camera gaps clear partial dwell and armed drag; cooldown or loss cannot cause repeated same-target dwell actions.
- Section 4: live locks require consecutive unsmoothed evidence with count, duration, displacement, and gap limits. Final missing/unstable readings invalidate earlier bounded-capture locks.
- Section 4: target inference no longer inflates detection confidence, and coordinate clamping cannot create false stable edge evidence.
- Section 3: signed calibration fitting rejects degenerate signals; visible targets, timed complete samples, mapped dispersion, and separate validation points gate profile saving.
- Section 3: live profiles require versioned camera/tracker-compatible validation evidence; failed calibration preserves the previous saved profile.
- Section 3: normalized macOS pointer conversion uses global display bounds rather than backing pixels, with simulated Retina-coordinate regression coverage.
- Section 2: live gaze remains active during speech; freshness and pending-action expiry use a monotonic clock, and confirmations are bound to app/window/target context.
- Section 2: a single camera worker owns open/read/close, invalidates locks on loss, and reports structured startup/read/shutdown failures.
- Section 1: live voice confirmation can now complete in the same process; successive commands can share one session.
- Section 1: listening and confirmation messages are delivered before blocking capture, with immediate JSONL logging.
- Section 1: punctuation, polite prefixes, bare gaze commands, low-confidence confirmations, and superseded pending commands are covered by regression tests.

## Product / Technical Risks

- **Eye tracking noise:** webcam-only gaze estimation may be too unstable for precise cursor control without a larger target model, dwell logic, or hardware assistance.
- **General-user scope risk:** "basic controls for everyone" can expand too quickly unless the first 10 command flows are kept explicit.
- **Unsafe command execution:** language models should not directly execute desktop actions; all action candidates need broker-side policy and confirmation rules.
- **Latency risk:** voice interaction will feel broken if transcription, intent parsing, and acknowledgement do not stay close to realtime.
- **Mode confusion:** voice and gaze can conflict if the system cannot clearly separate "selection", "navigation", and "command" states.
- **macOS adapter limitation:** the current macOS control slice supports `open_app`, `switch_app`, safe `press_key`, coarse `scroll`, notification-based `highlight_target`, point-based `focus_target`, and coordinate-based or normalized-gaze-based `click_target` / `double_click_target` / `right_click_target` / `drag_target`. Live pointer actions still depend on local macOS accessibility permissions, and target-ref-only grounding without normalized gaze or bounds is not implemented yet.
- **Voice-provider limitation:** a real macOS microphone-to-transcript path now exists, but it is macOS-only, bounded to final-transcript capture, and depends on microphone plus speech-recognition permissions.
- **Translation-provider limitation:** `translate_text` is recognized in the voice loop, but no local translation provider is configured yet.
- **Webcam-permission limitation:** the live webcam calibration and `gaze-live` paths need macOS camera permission for the active Python process; without that permission they recover with explicit structured errors.
- **Gaze-precision limitation:** the current webcam provider and inferencer are tuned for large targets and conservative dwell-based actions, not fine cursor replacement or dense gaze-only desktop control.
- **Calibration limitation:** acceptance thresholds are initial coarse-target engineering limits. Passing this check does not establish real task accuracy, drift resistance, or a probability of correct targeting. Only the primary display is supported; physical camera/display/seating changes require recalibration.
- **Gaze-robustness limitation:** Haar-based face/eye detection and thresholded pupil estimation are heuristic; lighting, glasses glare, and pose can degrade performance.
- **Gaze-control limitation:** `gaze-live` now has continuous `cursor` follow plus explicit dwell-triggered move/click/right-click/double-click/drag modes, but they are still coarse tester modes and can misfire on small targets or unstable lighting.
- **Cursor-follow limitation:** the new `cursor` mode uses smoothing, deadzone, edge padding, and max-step heuristics rather than true calibrated eye-tracker dynamics, so some per-machine tuning may still be needed.
- **Fusion-grounding limitation:** the multimodal fusion loop currently grounds deictic voice commands against the latest fresh gaze target window rather than a richer desktop-semantic target set.
- **Live-fusion orchestration limitation:** `fusion-live` maintains concurrent camera capture across repeated commands and confirmations, but speech still uses fixed windows and gaze binding happens at command finalization.
- **Focus-identity limitation:** foreground app/window checks do not identify exact editable fields or detect a transient focus change that returns before the next probe. Accessibility target identity remains needed.
- **Fusion-threshold limitation:** fused confidence uses fixed heuristic weights and a static clarification threshold; those values have not been tuned against real user-session data yet.
- **Overlay limitation:** the current productization layer exposes a terminal-rendered overlay snapshot rather than a native always-on-top macOS overlay.
- **Packaging limitation:** the repo now has a local CLI entry point and a smoke wrapper, but it is not packaged as a macOS app bundle or installer.

## Failed Approaches

- The first Phase 9 workflow used `macos-14`, which GitHub Actions retired on November 2, 2026. Those jobs were cancelled before setup; the workflow now uses the supported `macos-15` label. Check the [runner retirement announcements](https://github.com/actions/runner-images/issues/13518) when changing CI images.
- A direct Swift bridge to `Speech.framework` was rejected in this environment because the local Swift compiler and installed SDK do not match closely enough to build Foundation-backed framework imports reliably; the Phase 6 live speech path uses Objective-C plus `clang` instead.
