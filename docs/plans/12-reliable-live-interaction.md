# 12: Reliable Live Interaction

**Status:** completed
**Goal:** Repair the voice and webcam control foundations before measuring performance on the user's Mac.

## Delivery
- Work on `codex/interaction-reliability`.
- Implement, review, test, commit, and push each section before beginning the next.
- Keep existing dry-run defaults and machine-readable CLI output.
- Record hardware-dependent validation separately from automated correctness checks.

## Sections
- [x] [12-1-voice-session](12-1-voice-session.md): understandable commands, immediate feedback, and complete live follow-ups.
- [x] [12-2-live-context](12-2-live-context.md): real time, bounded pending actions, actual application context, and fresh live fusion.
- [x] [12-3-calibration-quality](12-3-calibration-quality.md): visible calibration, signed fitting, independent validation, and persisted quality.
- [x] [12-4-tracking-recovery](12-4-tracking-recovery.md): tracking-loss recovery, dwell rearming, and meaningful target-lock evidence.
- [x] [12-5-build-and-ci](12-5-build-and-ci.md): reproducible dependencies, packaged native helpers, and Linux/macOS CI.

## Boundaries
- This phase repairs the source currently on GitHub; the newer native application described in the review is not in this tree.
- Tracker comparisons and measured webcam accuracy require the actual Mac, camera, and user.
- Accessibility target snapping, a contextual LLM, and a production native shell remain separate milestones.
- Ordinary explicit actions should avoid redundant approval once their actual target/context can be verified. Coarse screen regions are not semantic controls.

## Plan Review
- Repair event delivery and lifecycle before adding concurrent inputs.
- Inject clocks and providers so timeouts, focus changes, and sensor loss can be tested without OS actions.
- Use regression tests for the reproduced failure paths; never label synthetic tests as measured gaze accuracy.

## Completion evidence

- Each implementation section was reviewed, committed, and pushed before the next began.
- The final clean-environment suite contains 476 passing tests. [CI run 42963494427](https://github.com/2Mars4096/interaction/actions/runs/42963494427) passed all eight Linux/macOS jobs, including installed-wheel native helper compilation on macOS ARM64.
- Review records for all five sections are in `docs/reviews/`. Hardware evaluation and the product boundaries above remain separate work.
