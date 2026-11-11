# Section 3: Calibration Quality Review

**Gate:** passed after fixes
**Plan:** [12-3-calibration-quality](../plans/12-3-calibration-quality.md)

## Findings and fixes
- A cancellation queued during the final camera read could previously miss the last UI pump; the CLI now processes pending UI events before accepting the result. Both existing-profile and no-profile regressions pass.
- Target means alone conceal jitter amplified by fitting. Aggregates now report timed sampling and dispersion; fitting and validation apply mapped screen-space dispersion gates.
- Normalized pointer conversion used backing pixel dimensions. It now uses Quartz global display bounds, consistent with the calibration reference coordinate domain.

## Evidence
- Full suite: 285 tests passed; `git diff --check` passed.
- Signed axes, insufficient signal, noisy/partial capture, independent validation without refitting, malformed/legacy/incompatible profiles, and atomic replacement failure are covered.
- Headless UI tests verify target geometry, main-thread use, responsive cancellation, partial-open cleanup, and no invisible fallback.
- CLI tests verify five visible fitting targets, four separate validation targets, resource cleanup, and byte-for-byte preservation of the previous profile on failure.
- Coordinate regression uses simulated Retina backing dimensions. API basis: [Apple CGDisplayBounds](https://developer.apple.com/documentation/coregraphics/cgdisplaybounds(_:)) and [Quartz display services](https://developer.apple.com/documentation/coregraphics/quartz-display-services).

## Limits
- Default error/dispersion gates are coarse engineering checks, not established webcam performance.
- Real macOS rendering, Retina behavior, permission prompts, and calibration accuracy remain hardware checks.
- The current pupil signal and primary-display scope remain unchanged; physical camera/display/seating changes require recalibration.
