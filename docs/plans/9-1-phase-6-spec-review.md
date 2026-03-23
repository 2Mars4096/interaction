# 9-1: Phase 6 Spec Review

**Parent:** [9-phase-6-mvp-readiness](9-phase-6-mvp-readiness.md)
**Status:** completed
**Goal:** Patch the MVP-readiness plan until the live-input scope, acceptance targets, and tester flow are concrete.

## Tasks
- [x] 1. Define the live voice scope
  - [x] 1-1. Choose the first real provider path
  - [x] 1-2. Decide whether streaming is required in this patch
- [x] 2. Define MVP acceptance targets
  - [x] 2-1. Latency and accidental-action targets
  - [x] 2-2. Tester-facing readiness conditions
- [x] 3. Define the local tester flow
  - [x] 3-1. Permissions and setup expectations
  - [x] 3-2. CLI or harness expectations
- [x] 4. Patch roadmap docs if needed

## Decisions
- The first real microphone path will use a small Objective-C bridge to macOS `Speech.framework` and `AVFoundation`, because the local Swift toolchain currently mismatches the installed SDK while `clang` can compile Objective-C framework clients successfully.
- This patch does not require realtime partial transcripts. A bounded push-to-talk style capture window with a final transcript is enough to make the MVP testable.
- MVP readiness now means: real microphone transcription on macOS, live command execution through the existing broker and adapter stack, explicit readiness targets, and a documented tester flow.
- Acceptance targets should be conservative and measurable: no medium-risk auto-execution without confirmation, clear recovery on missing permissions, and documented latency/accuracy goals for the first tester loop.

## Notes
- The first live tester path can still remain dry-run by default for safety.
