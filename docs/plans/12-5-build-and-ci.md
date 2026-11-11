# 12-5: Reproducible Build and CI

**Parent:** [12-reliable-live-interaction](12-reliable-live-interaction.md)
**Status:** in-progress
**Goal:** Make supported development installs, distributed speech resources, and platform build checks repeatable before hardware evaluation.

## Tasks
- [x] Bound runtime dependency compatibility, including OpenCV <5, and record a hashed development/test dependency baseline.
- [x] Include the Objective-C helper in wheel and source distributions; verify installed-package resources outside the checkout.
- [x] Bound helper compilation and capture subprocess time, report unsupported/missing/build/runtime failures, and validate native payloads.
- [x] Cache helper builds by source/platform identity and preserve the previous binary on failed rebuilds.
- [x] Clean native recording files on failure and cancel timed-out recognition tasks.
- [x] Add Linux/macOS CI with Python version coverage, tests, critical lint, distribution/resource checks, and native compile-only verification.
- [x] Run clean local dependency/build/install checks and independent source review.
- [ ] Commit and push; inspect available CI results.

## Decisions and plan review
- Lock the known working runtime baseline rather than silently taking a new major OpenCV release that removes used APIs.
- Keep project compatibility ranges separate from the hashed development baseline; document the Python versions actually exercised by CI.
- Native CI compiles the helper without recording audio, opening cameras, requesting permissions, or moving the pointer.
- Native microphone, camera, display, and user-performance checks remain hardware evaluation tasks.
- Preserve command/dictation text and unknown confidence semantics while rejecting malformed helper output.
- Review requires a wheel built from the sdist, installed-entrypoint checks outside the checkout, and no unpinned isolated backend downloads.
- Parent-owned temporary recording storage must be cleaned even when the helper times out or is interrupted.
- Package and embed microphone/speech usage descriptions in the native executable; Apple's Speech/AVFoundation authorization APIs require these keys.

## Verification

- Full clean-environment suite: 476 passed; critical lint, dependency metadata, and diff checks passed.
- Built the wheel from the sdist with the locked build tools; both archives passed resource/baseline verification.
- Installed the wheel into a separate virtual environment outside the checkout; CLI, imports, packaged resources, and Haar classifiers passed on Linux/Python 3.12.
- Independent source gate: [Section 5 review](../reviews/2026-11-11-section-5-build-and-ci.md).
- Remote macOS compilation is pending the first CI run; local checks do not claim native execution or hardware accuracy.
