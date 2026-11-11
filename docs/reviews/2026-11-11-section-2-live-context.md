# Section 2: Live Context Review

**Gate:** passed after fixes
**Plan:** [12-2-live-context](../plans/12-2-live-context.md)

## Findings and fixes
- A worker revision change could invalidate the newly received observation by setting a later loss watermark; invalidation now uses the snapshot's original timestamp before applying it.
- The final result could retain a previous successful target after loss; unavailable gaze now removes stale target/observation fields.
- Continuous voice sessions recover after a changed-context command is cancelled, allowing a repeat without restarting.

## Evidence
- Full automated suite: 163 tests passed.
- Actual background worker integration with fake camera/speech verifies continued capture during command and confirmation, loss/reacquisition cancellation, final loss reporting, context changes, and camera cleanup.
- Deterministic clocks cover expiry, timestamp ordering, target revision changes, point drift, and app/window changes.
- Probe tests cover valid, unavailable, malformed, timed-out, and unsupported-platform context. CLI tests isolate host desktop focus.
- `git diff --check` passed.

## Limits
- Native macOS permission behavior and camera performance require hardware testing.
- Gaze binds at command finalization; exact speech-onset alignment and exact editable-field identity require further integration.
- A camera backend stuck inside a native read may outlive the bounded shutdown timeout; this is surfaced explicitly rather than silently reported as successful cleanup.
