# 9-2: macOS Live Speech Provider

**Parent:** [9-phase-6-mvp-readiness](9-phase-6-mvp-readiness.md)
**Status:** completed
**Goal:** Add the first real macOS microphone-to-transcript provider and connect it to the existing voice command runtime.

## Tasks
- [x] 1. Add the native bridge source and build path
  - [x] 1-1. Check in the Objective-C helper source
  - [x] 1-2. Compile it on demand into the local runtime directory
- [x] 2. Add the Python-side provider wrapper
  - [x] 2-1. Capture structured success and failure payloads
  - [x] 2-2. Handle microphone or speech-permission denial cleanly
- [x] 3. Integrate the live provider into the CLI and voice loop
  - [x] 3-1. Add a live voice command path
  - [x] 3-2. Keep dry-run as the default execution mode
- [x] 4. Add tests around helper management and provider parsing

## Decisions
- The real macOS speech path now compiles a tiny Objective-C helper on demand into `.interaction/bin/` instead of relying on the mismatched local Swift toolchain.
- The first live provider uses bounded microphone capture plus a final transcript rather than streaming partials.
- Permission denial is treated as a structured recovery outcome and is now covered by both live smoke behavior and automated tests.

## Notes
- The provider should fail with explicit structured errors instead of stack traces when permissions or framework support are missing.
