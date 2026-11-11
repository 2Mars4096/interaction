# Section 5: Build and CI Review

**Source gate:** passed after fixes
**Platform gate:** passed on Linux and macOS ARM64
**Plan:** [12-5-build-and-ci](../plans/12-5-build-and-ci.md)

## Findings and fixes

- Temporary recording directory creation/cleanup errors previously escaped the speech error contract. They now return a structured storage error.
- Two processes could race while publishing the first helper cache entry. A losing publisher now reuses the exact verified winning binary/manifest pair.
- JSON overflow numbers such as `1e309` could retain non-finite values in error metadata. Parsing rejects them before they enter the result payload.
- Distribution checks require the Objective-C source, permission plist, dependency constraints, and hashed baseline in the appropriate archives. The wheel is built from the sdist.
- The first published workflow used the retired `macos-14` image and its Mac jobs were cancelled before setup. The matrix now uses `macos-15`, following the [official retirement announcement](https://github.com/actions/runner-images/issues/13518).

## Evidence

- Full suite: **476 passed** in a clean environment using `requirements-dev.lock`.
- Independent reviewer: 123 focused speech/provider/voice/command/distribution tests passed.
- Critical Ruff checks, `pip check`, and `git diff --check` passed.
- `python -m build --no-isolation` produced the sdist and its wheel with locked build tools; archive resource/baseline checks passed.
- The wheel passed an isolated install outside the checkout on Linux/Python 3.12.14: CLI entry point, installed import origins, exact native resources, and all required OpenCV 4.12 Haar classifiers.
- [First CI run](https://github.com/2Mars4096/interaction/actions/runs/42963333479): all three Linux Python-version test jobs and the Linux distribution job passed. The four Mac jobs were cancelled before setup on the retired runner, prompting the matrix correction above.
- [Corrected CI run](https://github.com/2Mars4096/interaction/actions/runs/42963494427) at `3c94d344840393937107d2d8d7416fc78b901524`: all eight jobs passed. Tests ran on Linux and macOS with Python 3.11, 3.12, and 3.13; both distribution jobs passed.
- The macOS 15 ARM64 distribution job explicitly reported `native_compile: passed` for the installed helper and verified cache reuse. The macOS Python 3.13 test log reported `476 passed`.
- Reviewed finite build/capture deadlines, source/platform cache identity, manifest verification, rollback, interruption cleanup, literal transcript preservation, and strict native payload metadata.
- CI defines six Linux/macOS test jobs across Python 3.11–3.13 and two distribution jobs. The macOS distribution job compiles the installed helper and verifies cache reuse without running capture.

## Limits

- Native compilation is verified on the macOS ARM64 CI runner; the local development environment is Linux.
- Automated verification does not exercise microphone/camera permissions, speech quality or latency, desktop actions, webcam accuracy, or user performance.
- The speech provider still records bounded turns. Streaming recognition, semantic Accessibility targets, and a native app bundle remain separate work.
