# 8-3: Packaging, Calibration, and Persistence

**Parent:** [8-phase-5-productization](8-phase-5-productization.md)
**Status:** completed
**Goal:** Make the product repeatable for local use by persisting key state and improving setup/run ergonomics.

## Tasks
- [x] 1. Persist important state
  - [x] 1-1. Calibration data
  - [x] 1-2. User settings and safe defaults
- [x] 2. Improve packaging and run flows
  - [x] 2-1. Developer setup
  - [x] 2-2. Tester setup
- [x] 3. Add tests or smoke checks for persistence and startup behavior

## Decisions
- Runtime state now persists inside a local `.interaction/` directory with separate paths for settings, session logs, and calibration profiles.
- The product now exposes a lightweight CLI entry point plus a repo-local Phase 5 smoke wrapper so repeated testing does not depend on custom shell snippets.
- The persisted settings currently cover safe defaults such as dry-run mode, gaze context window, and dwell timing.

## Notes
- This work should reduce friction for repeated Phase 2 and Phase 3 testing.
