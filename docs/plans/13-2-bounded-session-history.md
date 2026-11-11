# 13-2: Bounded Session History

**Parent:** [13-responsive-live-sessions](13-responsive-live-sessions.md)
**Status:** completed
**Goal:** Keep continuous sessions' memory and final JSON bounded while preserving complete event replay.

## Tasks
- [x] Bound live feedback objects, serialized events, overlay snapshots, and capture summaries with a configurable positive history limit.
- [x] Report total/retained/dropped counts so truncation is explicit and session counters remain exact.
- [x] Keep incremental JSONL logging complete; do not re-log retained events when building the final payload.
- [x] Ensure repeated silence and streaming-style partial feedback cannot accumulate unbounded retained entries.
- [x] Test eviction order, bounded voice/fusion sessions, complete replay, and backward compatibility for short sessions; review before commit/push.

## Decisions
- Bound only live-session retained history. Scripted smoke and explicit full replay retain their existing behavior.
- The complete event journal is the audit record; stdout contains the most recent retained events and capture summaries.
- Default retention is 200 entries per collection. `--history-limit` is available on voice-live, fusion-live, gaze-live, and gaze-calibrate because all four share the live event sink.
- Shared interaction state already retains at most 12 entries per modality. Pending actions, overlay state, and the camera's current snapshot remain independent of output retention.
- This bounds retained entry counts; per-event protocol byte limits are covered by section 13-3.

## Verification
- 525 tests passed, including 16 retention regressions; critical Ruff and diff checks passed.
- Independent review confirmed output retention is separate from pending actions and current gaze state.
- Review: [13-2 bounded history](../reviews/2026-11-11-13-2-bounded-session-history.md).
