# 12-3: Calibration Quality and Visible Targets

**Parent:** [12-reliable-live-interaction](12-reliable-live-interaction.md)
**Status:** completed
**Goal:** Save usable live profiles only after stable fitting samples and independent accuracy checks against visible reference points.

## Tasks
- [x] Fit signed per-axis least-squares mappings; reject degenerate and non-finite calibration input.
- [x] Display actual on-screen targets and immediate progress; support Escape/close cancellation and guaranteed cleanup.
- [x] Require sufficient stable samples at every fitting target, with real sampling intervals and honest dispersion evidence.
- [x] Measure prediction error on a separate set of validation targets without fitting on those measurements.
- [x] Persist versioned validation quality and camera identity; reject legacy, malformed, incompatible, or unvalidated live profiles.
- [x] Add deterministic regression coverage for reversed axes, constant input, partial/noisy samples, validation failures, persistence, and CLI lifecycle.
- [x] Review, document thresholds and hardware limits, commit, and push.

## Decisions and plan review
- Keep the current tracking signal and per-axis affine model in this section; changing trackers requires new hardware measurements.
- Five fitting targets and four different validation targets separate fitting from the acceptance check.
- Measure errors before output clamping. Defaults are coarse-target engineering gates, not evidence of achieved cursor precision.
- A visible calibration screen is required for live calibration; unavailable UI returns a structured error rather than silently collecting unknown reference positions.
- Legacy four-number profiles remain readable for scripted demos but cannot enable live webcam control without fresh validation.
- Preserve the last saved valid profile if a new calibration fails or is cancelled; save successful results atomically.
- Review requires mapped screen-space dispersion so fitted scales cannot hide jitter, camera plus tracker/signal compatibility, and proof validation never changes fitting coefficients.
- Reference points and live normalized pointer positions address the primary display only. Multi-display calibration remains separate work.
- Replace framebuffer-pixel conversion with Quartz global display bounds so Retina pointer coordinates match the calibration screen's logical bounds.
- Initial coarse gates: mean held-out error <=0.12, worst error <=0.20, mapped per-axis standard deviation <=0.05, and raw fitting span >=0.005 on both axes. These are explicit engineering limits pending hardware measurement.
