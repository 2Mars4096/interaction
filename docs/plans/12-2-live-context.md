# 12-2: Time and Live Context

**Parent:** [12-reliable-live-interaction](12-reliable-live-interaction.md)
**Status:** completed
**Goal:** Bind voice actions to fresh gaze and the actual foreground context throughout a live session.

## Tasks
- [x] Use injected monotonic time for gaze observations and pending action expiry.
- [x] Cancel pending actions after context change, timeout, tracking loss, or target change.
- [x] Collect actual foreground application/window context with a bounded macOS probe; represent unavailable context honestly.
- [x] Keep one camera-owning worker active through speech and confirmation; consume timestamped recent stable observations.
- [x] Release worker/camera resources on completion, errors, and interruption.
- [x] Cover elapsed time, focus changes, stale targets, worker failures, and cleanup with deterministic tests.
- [x] Review, document limitations, commit, and push.

## Decisions and plan review
- A real clock alone would break the old sequential four-second voice flow; sensing must continue during capture.
- Select the recent stable gaze target at command finalization and bind any pending action to it. Future onset-based binding needs actual streaming audio timestamps and is not simulated here.
- One worker owns camera open/read/close. The main thread interprets commands and executes actions.
- Existing scripted `advance()` calls remain an explicit simulation facility; real live input never fabricates elapsed time from a nominal frame rate.
- Application/window checks are an initial boundary; exact editable-field identity requires the later Accessibility target model.
