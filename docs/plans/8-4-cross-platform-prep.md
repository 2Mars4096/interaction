# 8-4: Cross-Platform Prep

**Parent:** [8-phase-5-productization](8-phase-5-productization.md)
**Status:** completed
**Goal:** Prepare the architecture and adapter boundaries so the macOS-first system can later grow into a cross-platform product without a full redesign.

## Tasks
- [x] 1. Audit macOS-specific assumptions
  - [x] 1-1. Separate platform-agnostic broker logic from adapter logic
  - [x] 1-2. Document macOS-only surfaces
- [x] 2. Define cross-platform adapter seams
  - [x] 2-1. Window/app primitives
  - [x] 2-2. Input primitives
- [x] 3. Update docs and architecture notes for the expansion path

## Decisions
- Platform adapters now expose explicit capability descriptions so product code can reason about supported actions and macOS-only limitations without hard-coding those assumptions elsewhere.
- The broker and runtime layers remain platform-agnostic; macOS-specific planning and permission notes stay inside the adapter capability surface and architecture docs.
- Cross-platform prep in this phase is limited to seam-hardening and documentation, not new platform implementations.

## Notes
- This subplan is preparation work, not a commitment to implement Windows/Linux immediately.
