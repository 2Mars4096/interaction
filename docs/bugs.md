# Known Issues & Failed Approaches

## Open Bugs

- None recorded yet.

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
- **Gaze-robustness limitation:** Haar-based face/eye detection and thresholded pupil estimation are heuristic; lighting, glasses glare, and pose can degrade performance.
- **Gaze-control limitation:** `gaze-live` now has continuous `cursor` follow plus explicit dwell-triggered move/click/right-click/double-click/drag modes, but they are still coarse tester modes and can misfire on small targets or unstable lighting.
- **Cursor-follow limitation:** the new `cursor` mode uses smoothing, deadzone, edge padding, and max-step heuristics rather than true calibrated eye-tracker dynamics, so some per-machine tuning may still be needed.
- **Fusion-grounding limitation:** the multimodal fusion loop currently grounds deictic voice commands against the latest fresh gaze target window rather than a richer desktop-semantic target set.
- **Live-fusion orchestration limitation:** the current `fusion-live` path captures gaze context first and speech second; it is a real MVP demo path, but not yet a continuous concurrent multimodal session loop.
- **Fusion-threshold limitation:** fused confidence uses fixed heuristic weights and a static clarification threshold; those values have not been tuned against real user-session data yet.
- **Overlay limitation:** the current productization layer exposes a terminal-rendered overlay snapshot rather than a native always-on-top macOS overlay.
- **Packaging limitation:** the repo now has a local CLI entry point and a smoke wrapper, but it is not packaged as a macOS app bundle or installer.

## Failed Approaches

- A direct Swift bridge to `Speech.framework` was rejected in this environment because the local Swift compiler and installed SDK do not match closely enough to build Foundation-backed framework imports reliably; the Phase 6 live speech path uses Objective-C plus `clang` instead.
