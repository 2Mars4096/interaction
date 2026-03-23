# 5-3: Transcription and Intent Loop

**Parent:** [5-phase-2-voice-mvp](5-phase-2-voice-mvp.md)
**Status:** completed
**Goal:** Convert voice turns into transcript segments, normalized intents, proposals, and broker-ready decisions.

## Tasks
- [x] 1. Integrate a transcription provider boundary
  - [x] 1-1. Stream partial and final transcripts
  - [x] 1-2. Normalize transcript segments into contracts
- [x] 2. Implement voice-only intent handling
  - [x] 2-1. Map simple commands into normalized intents
  - [x] 2-2. Route unsupported or ambiguous requests into clarify/reject flows
- [x] 3. Wire the voice loop into the broker
  - [x] 3-1. Produce proposals
  - [x] 3-2. Consume broker decisions
- [x] 4. Add tests for the end-to-end voice-only path

## Decisions
- The Phase 2 interpreter is intentionally pattern-based and bounded rather than LLM-driven.
- Deictic commands without gaze grounding explicitly clarify instead of guessing.
- `translate_text` is recognized but recovers early because no local translation provider is configured yet.

## Notes
- This phase should avoid dependence on gaze for success on simple commands.
