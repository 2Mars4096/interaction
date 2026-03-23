# 7-3: Confidence, Clarification, and Evaluations

**Parent:** [7-phase-4-multimodal-fusion](7-phase-4-multimodal-fusion.md)
**Status:** completed
**Goal:** Add the decision logic and evaluation harnesses needed to judge whether fusion is actually improving the interaction loop.

## Tasks
- [x] 1. Implement confidence aggregation
  - [x] 1-1. Combine transcript confidence, gaze confidence, and target confidence
  - [x] 1-2. Map combined confidence to allow/confirm/clarify defaults
- [x] 2. Implement clarification flows
  - [x] 2-1. Ask follow-up questions
  - [x] 2-2. Present likely targets or options
- [x] 3. Implement evaluation harnesses
  - [x] 3-1. Record success, clarification, and accidental-action metrics
  - [x] 3-2. Support repeatable task runs for the showcase set
- [x] 4. Add tests around confidence thresholds and clarify behavior

## Decisions
- Fused command confidence now combines transcript, gaze-observation, and grounded-target confidence before a deictic action proposal is allowed to proceed.
- Low-confidence deictic grounding clarifies instead of confirming, and clarification messages name likely recent targets rather than silently failing.
- Phase 4 evaluation evidence consists of repeatable fusion smoke runs plus automated metrics and targeted regression tests for stale-target and low-confidence cases.

## Notes
- This subplan should produce the first serious evidence on whether multimodal fusion is worth the complexity.
