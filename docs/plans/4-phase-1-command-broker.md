# 4: Phase 1 Command Broker and macOS Control Base

**Status:** completed
**Goal:** Turn the contract layer into a safe execution core by defining the broker policy engine and the first bounded macOS action slice.

## Tasks
- [x] 1. Finalize the phase spec and patch it before code
  - [x] 1-1. Complete [4-1-phase-1-spec-review](4-1-phase-1-spec-review.md)
- [x] 2. Build the command broker core
  - [x] 2-1. Complete [4-2-command-broker-core](4-2-command-broker-core.md)
- [x] 3. Build the first macOS control slice
  - [x] 3-1. Complete [4-3-macos-control-slice](4-3-macos-control-slice.md)
- [x] 4. Close the phase with a review gate
  - [x] 4-1. Complete [4-4-phase-1-code-review-gate](4-4-phase-1-code-review-gate.md)

## Decisions
- Phase 1 is the first phase that touches live execution, so safety and observability matter more than breadth.
- The broker should own risk policy, not the platform adapter.
- The first bounded command set and default broker decisions are recorded in `docs/command-vocabulary.md`.

## Notes
- The user intends to test through Phase 3, so Phase 1 should keep the first executable slice as narrow and demoable as possible.
- Phase 1 is now complete: broker policy exists, the first bounded macOS adapter exists, and the review gate passed.
