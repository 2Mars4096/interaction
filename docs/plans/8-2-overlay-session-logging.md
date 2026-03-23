# 8-2: Overlay and Session Logging

**Parent:** [8-phase-5-productization](8-phase-5-productization.md)
**Status:** completed
**Goal:** Build the user-facing overlay and logging surfaces needed for trust, replay, and repeated testing.

## Tasks
- [x] 1. Implement the overlay surface
  - [x] 1-1. Show listening/interpreting/confirming/executing state
  - [x] 1-2. Show transcript, target, and broker decision feedback
- [x] 2. Implement session logging
  - [x] 2-1. Log transcript, grounded targets, proposals, broker decisions, and results
  - [x] 2-2. Support replay or inspection for debugging
- [x] 3. Add tests or harnesses for overlay state transitions and logging integrity

## Decisions
- The Phase 5 overlay is now a structured `OverlayState` plus a terminal renderer so runtime state is visible without binding the project to a native macOS UI toolkit.
- Session logging now writes structured JSONL records and supports replay into overlay snapshots for debugging and demos.
- Productization smokes now run through the same logging and overlay surfaces as the CLI, rather than bespoke one-off printing paths.

## Notes
- This subplan is about trust and observability as much as UI.
