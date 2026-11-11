# 12-1: Voice Commands and Live Session

**Parent:** [12-reliable-live-interaction](12-reliable-live-interaction.md)
**Status:** completed
**Goal:** Let users issue ordinary speech commands and finish any required follow-up in the same visible session.

## Tasks
- [x] Normalize command punctuation and polite prefixes without changing dictated text.
- [x] Resolve short/deictic commands before generic application names; support bare click/right click/double click.
- [x] Respect low recognition confidence for commands and confirmations; cancellation remains available.
- [x] Clear superseded pending actions when a new command arrives.
- [x] Stream live feedback to stderr while preserving final JSON on stdout.
- [x] Capture bounded follow-ups in voice-live and offer a persistent multi-turn session.
- [x] Cover actual CLI lifecycle and event ordering with fake input providers.
- [x] Review, update tracking/user docs, commit, and push.

## Decisions
- Unknown/coarse pointer targets retain the broker's confirmation boundary until real target semantics exist.
- A continuous CLI session is an orchestration improvement; the current native recording provider remains bounded capture until the streaming provider milestone.
- Keep transcript payloads intact, including whitespace and punctuation in dictation.

## Verification
- Full suite: 105 passed before the final fusion prompt-order regression; all 14 live voice/fusion tests passed afterward (106 tests total).
- Independent review found silent turns ending continuous sessions; patched and covered silence/retry/exhaustion.
- `git diff --check` passed. Native microphone behavior still requires macOS testing.
