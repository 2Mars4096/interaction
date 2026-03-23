# 5-1: Phase 2 Spec Review

**Parent:** [5-phase-2-voice-mvp](5-phase-2-voice-mvp.md)
**Status:** completed
**Goal:** Patch the voice-phase plan until capture mode, transcript UX, and confirmation behavior are concrete.

## Tasks
- [x] 1. Decide the initial interaction style
  - [x] 1-1. Choose always-listening, push-to-talk, or hybrid
  - [x] 1-2. Define how the listening state is shown
- [x] 2. Finalize the voice-only command coverage
  - [x] 2-1. Map commands that do not need gaze grounding
  - [x] 2-2. Define how dictation differs from command mode
- [x] 3. Patch the contracts and docs if needed
- [x] 4. Freeze the acceptance criteria for Phase 2

## Decisions
- Push-to-talk is the Phase 2 default. The architecture should stay compatible with a later hybrid model, but always-listening is not the initial implementation.
- The first feedback surface for Phase 2 is explicit textual state plus transcript updates; richer overlay polish stays later.
- Phase 2 voice-only coverage includes `open_app`, `switch_app`, `scroll`, safe `press_key`, explicit `type_text`, explicit `translate_text`, plus spoken confirmation and cancellation.
- Target-dependent commands such as `click this` and `open this` should not guess without gaze grounding; they should clarify.
- Dictation is limited in Phase 2 to explicit command-style entry such as `type ...` or `dictate ...`; there is no freeform ambient dictation mode yet.
- Because the repository does not yet commit to a microphone/STT provider, Phase 2 acceptance is defined around a realtime-like transcript event pipeline with partial/final segments, push-to-talk turn handling, command parsing, confirmation/cancellation, and broker/macOS dry-run integration.

## Notes
- Voice UX ambiguity should be resolved in plan review, not discovered late in implementation.
