# Section 5: Build and CI Review

**Source gate:** passed after fixes
**Platform gate:** pending first published CI run
**Plan:** [12-5-build-and-ci](../plans/12-5-build-and-ci.md)

## Findings and fixes

- Temporary recording directory creation/cleanup errors previously escaped the speech error contract. They now return a structured storage error.
- Two processes could race while publishing the first helper cache entry. A losing publisher now reuses the exact verified winning binary/manifest pair.
- JSON overflow numbers such as `1e309` could retain non-finite values in error metadata. Parsing rejects them before they enter the result payload.
- Distribution checks require the Objective-C source, permission plist, dependency constraints, and hashed baseline in the appropriate archives. The wheel is built from the sdist.

## Evidence

- Full suite: **476 passed** in a clean environment using `requirements-dev.lock`.
- Independent reviewer: 123 focused speech/provider/voice/command/distribution tests passed.
- Critical Ruff checks, `pip check`, and `git diff --check` passed.
- `python -m build --no-isolation` produced the sdist and its wheel with locked build tools; archive resource/baseline checks passed.
- The wheel passed an isolated install outside the checkout on Linux/Python 3.12.14: CLI entry point, installed import origins, exact native resources, and all required OpenCV 4.12 Haar classifiers.
- Reviewed finite build/capture deadlines, source/platform cache identity, manifest verification, rollback, interruption cleanup, literal transcript preservation, and strict native payload metadata.
- CI defines six Linux/macOS test jobs across Python 3.11–3.13 and two distribution jobs. The macOS distribution job compiles the installed helper and verifies cache reuse without running capture.

## Limits

- Remote macOS compilation results must be checked after publishing the workflow. No native compilation is claimed by the Linux source gate.
- Automated verification does not exercise microphone/camera permissions, speech quality or latency, desktop actions, webcam accuracy, or user performance.
- The speech provider still records bounded turns. Streaming recognition, semantic Accessibility targets, and a native app bundle remain separate work.
