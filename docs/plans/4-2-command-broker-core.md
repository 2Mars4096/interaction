# 4-2: Command Broker Core

**Parent:** [4-phase-1-command-broker](4-phase-1-command-broker.md)
**Status:** completed
**Goal:** Implement the policy and dispatch core that turns intents and proposals into broker decisions and auditable execution requests.

## Tasks
- [x] 1. Create the broker module structure
  - [x] 1-1. Add control/broker package layout
  - [x] 1-2. Add policy and dispatch interfaces
- [x] 2. Implement broker policy evaluation
  - [x] 2-1. Map risk levels and action types to default decisions
  - [x] 2-2. Add confirmation and clarification prompt generation
- [x] 3. Implement audit-friendly execution preparation
  - [x] 3-1. Convert proposals into execution requests
  - [x] 3-2. Preserve rationale, broker reason, and session context
- [x] 4. Add tests
  - [x] 4-1. Policy table coverage
  - [x] 4-2. Invalid proposal and invalid decision cases
  - [x] 4-3. Serialization / round-trip coverage for broker outputs

## Decisions
- `ExecutionRequest` is part of the shared contract layer so platform adapters receive a typed broker-approved request.
- The broker owns a `confirm()` promotion step so later UX can turn confirm decisions into allow decisions explicitly.
- Action-specific policy overrides are kept small and readable in `BrokerPolicy`.

## Notes
- This subplan should stop short of platform-specific side effects. It should prepare and govern them.
