# 8-1: Phase 5 Spec Review

**Parent:** [8-phase-5-productization](8-phase-5-productization.md)
**Status:** completed
**Goal:** Patch the productization-phase plan until overlay scope, persistence requirements, and packaging boundaries are concrete.

## Tasks
- [x] 1. Define the overlay scope
  - [x] 1-1. Determine which states must be visible at all times
  - [x] 1-2. Determine which controls belong in the overlay
- [x] 2. Define persistence requirements
  - [x] 2-1. Calibration persistence
  - [x] 2-2. Session logs and replay scope
- [x] 3. Define packaging expectations for macOS
  - [x] 3-1. Local developer flow
  - [x] 3-2. Repeatable tester flow
- [x] 4. Patch docs and acceptance criteria if needed

## Decisions
- Phase 5 will ship a structured overlay-state model plus a terminal renderer rather than a native GUI overlay. That keeps observability real without committing to a macOS-specific UI framework yet.
- Persistent runtime data will live in a local `.interaction/` workspace directory by default and will include session logs, replayable event history, calibration profiles, and user settings.
- Packaging will focus on a lightweight local CLI entry point so both developer and tester flows can run the same smokes and replay tools repeatably on macOS.
- Cross-platform prep in this phase will harden adapter seams and capability reporting, not implement Windows or Linux adapters yet.

## Notes
- Productization should not become a stealth rewrite; it should stabilize the existing path.
