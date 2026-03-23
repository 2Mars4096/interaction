# 3: Runtime Contracts

**Status:** completed
**Goal:** Create the first typed schema package for multimodal observations, normalized intents, action proposals, broker decisions, and execution results.

## Tasks
- [x] 1. Add a shared contracts package
  - [x] 1-1. Create `src/interaction/contracts/__init__.py`
  - [x] 1-2. Create `src/interaction/contracts/models.py`
- [x] 2. Define runtime enums and shared models
  - [x] 2-1. Platform, session, risk, decision, status, and action enums
  - [x] 2-2. Transcript, gaze, grounded target, environment, and session models
  - [x] 2-3. Multimodal context, normalized intent, action proposal, broker decision, and execution result models
- [x] 3. Add policy-oriented validation
  - [x] 3-1. Gaze observations require a target or coordinates
  - [x] 3-2. L3 proposals must require confirmation
  - [x] 3-3. `allow` cannot auto-approve L3 proposals
  - [x] 3-4. `confirm` and `clarify` require prompts
- [x] 4. Add tests
  - [x] 4-1. Round-trip contract serialization
  - [x] 4-2. Validation failure coverage for broker policy constraints
- [x] 5. Sync docs and dependency metadata

## Decisions
- Use Pydantic v2 for the first contract layer to keep validation and serialization explicit.
- Start with a single `models.py` file instead of prematurely splitting many modules.
- Keep action names bounded and small; expand only when the broker and platform layers actually need more verbs.
- Encode a few high-signal safety constraints directly in model validation so obvious policy violations fail early.

## Notes
- These models are intentionally scaffold-grade. They define the shared language of the system, not the full runtime behavior.
- The next phase should build the command broker on top of these contracts instead of inventing parallel ad hoc data shapes.
