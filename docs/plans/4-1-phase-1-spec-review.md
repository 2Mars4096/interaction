# 4-1: Phase 1 Spec Review

**Parent:** [4-phase-1-command-broker](4-phase-1-command-broker.md)
**Status:** completed
**Goal:** Patch the Phase 1 plans until the action vocabulary, risk policy, and macOS execution boundaries are concrete enough to implement safely.

## Tasks
- [x] 1. Finalize the first command vocabulary
  - [x] 1-1. Choose the first 10 showcase commands
  - [x] 1-2. Mark which commands are `allow`, `confirm`, `clarify`, or `reject` by default
- [x] 2. Finalize the first macOS execution boundary
  - [x] 2-1. Define what counts as in-scope app/window/cursor actions
  - [x] 2-2. Define what stays out of scope for Phase 1
- [x] 3. Patch the contract surface if needed
  - [x] 3-1. Add or revise enums and fields required by the broker
  - [x] 3-2. Patch docs if policy assumptions change
- [x] 4. Record the patched phase scope in the relevant docs

## Decisions
- The first bounded command set is explicit rather than implied.
- Safe auto-execution is limited to low-risk navigation-style commands and a tiny allow-list of safe keys.
- Click, double-click, type, and translate actions confirm by default.
- `L3` actions reject by default in MVP.

## Notes
- No implementation should start until the review confirms the action set and confirmation rules are explicit.
