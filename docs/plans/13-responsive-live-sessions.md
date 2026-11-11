# 13: Responsive Live Sessions

**Status:** in-progress
**Goal:** Let voice and gaze stay available across successive commands, bound long-session memory, and add real streaming speech before hardware evaluation.

## Delivery
- Continue `codex/interaction-reliability` from the completed Phase 9 work.
- Implement, independently review, test, commit, and push each section before starting the next.
- Preserve dry-run defaults, bounded action policy, context checks, and one final JSON document on stdout.
- Keep actual Mac/camera/voice latency and accuracy measurements separate from automated correctness and native compilation.

## Sections
- [x] [13-1-continuous-fusion](13-1-continuous-fusion.md): repeated commands and same-session recovery while one camera worker remains active.
- [ ] [13-2-bounded-session-history](13-2-bounded-session-history.md): bounded memory and final payloads with complete incremental event logs.
- [ ] [13-3-streaming-speech](13-3-streaming-speech.md): live audio recognition, immediate transcript previews, bounded endpointing, and final-only execution.

## Plan review
- Repository plans through 12-5 are complete; these sections address the recorded live-fusion and speech-provider limitations.
- Two independent code audits recommend repeated fusion first because it reuses the existing tested session and camera seams.
- Keep the initial gaze-lock gate, then refresh gaze for each command; no stale gaze may block a spoken stop or silently authorize a pointer action.
- Bound continuous session history before adding streaming partial events. The JSONL journal remains the complete event record.
- Streaming is an explicit speech-mode option until live Mac evaluation; recorded capture remains available and never silently retries a command through a second provider.
- Accessibility target identity, contextual LLM interpretation, native app UI, and webcam tracker replacement remain separate milestones.
