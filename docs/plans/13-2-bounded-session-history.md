# 13-2: Bounded Session History

**Parent:** [13-responsive-live-sessions](13-responsive-live-sessions.md)
**Status:** not-started
**Goal:** Keep continuous sessions' memory and final JSON bounded while preserving complete event replay.

## Tasks
- [ ] Bound live feedback objects, serialized events, overlay snapshots, and capture summaries with a configurable positive history limit.
- [ ] Report total/retained/dropped counts so truncation is explicit and session counters remain exact.
- [ ] Keep incremental JSONL logging complete; do not re-log retained events when building the final payload.
- [ ] Ensure repeated silence and streaming-style partial feedback cannot grow memory without limit.
- [ ] Test eviction order, bounded voice/fusion sessions, complete replay, and backward compatibility for short sessions; review before commit/push.

## Decisions
- Bound only live-session retained history. Scripted smoke and explicit full replay retain their existing behavior.
- The complete event journal is the audit record; stdout contains the most recent retained events and capture summaries.
