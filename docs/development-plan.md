# Interaction — Development Plan

## 1. Vision

Build a system that lets a general computer user interact with a computer more directly through realtime voice and eye behavior.

- **Voice** carries intent: what the user wants done
- **Gaze** carries target and context: where the user is focused
- **The system** fuses both into safe computer actions

The project is not just "voice commands" and not just "eye tracking". The interesting product is the joint control loop between language, gaze, and execution.

## 2. Product Thesis

- Voice is high-bandwidth for intent, but weak at precise targeting.
- Gaze is low-friction for targeting, but weak at expressing complex intent.
- Combining them can reduce friction: "open this", "translate that", "reply here", "scroll slower".
- A command broker must sit between perception and execution so the system stays safe, explainable, and interruptible.

## 3. Product Decisions Locked In

- **Primary user:** general computer user, not a narrow specialist persona
- **Primary platform:** macOS
- **Platform strategy:** design abstractions so Windows/Linux can follow later
- **Tracking hardware:** normal webcam only in the first prototype
- **Initial demo style:** broad basic-control showcase, not a single-task niche workflow

These decisions bias the product toward low-friction setup and visible everyday usefulness, but they also impose harder constraints on robustness because webcam eye tracking is noisy and general-purpose control has wider surface area.

## 4. Core Principles

- **Realtime first:** latency matters more than model cleverness in the first MVP.
- **Safety first:** the model proposes actions; the broker decides whether to execute, confirm, or reject.
- **Graceful fallback:** if confidence is low, the system asks, highlights, or does nothing.
- **Observable behavior:** the user should be able to see what was heard, inferred, targeted, and executed.
- **Accessibility value:** the system should improve access, not only feel novel.
- **General-user usability:** the first experience must be understandable without training the user on a complex command language.

## 5. MVP Interaction Loop

1. User speaks a command.
2. System transcribes it in realtime.
3. System combines transcript with current gaze target and desktop context.
4. System proposes a bounded action.
5. Broker decides:
   - execute immediately for low-risk actions
   - confirm for medium/high-risk actions
   - reject or ask a clarifying question when ambiguous
6. System provides visual or spoken feedback.

## 6. MVP Capability Slice

The first implementation should demonstrate a coherent set of basic controls rather than a broad but shallow feature list.

### Core showcase actions

- focus or highlight the current gaze target
- scroll up, down, faster, slower
- click the highlighted target
- open or switch to a named app
- press common navigation keys such as `escape`, `enter`, `tab`, `back`
- type short dictated text into the active field
- translate selected text or dictated text into another language

The explicit first bounded command set is captured in `docs/command-vocabulary.md`.

### Explicitly out of scope for the first pass

- complex desktop automation workflows
- unrestricted shell or terminal execution
- sensitive account actions such as purchases, financial transfers, or password entry
- precision-grade cursor control that depends on dedicated eye-tracking hardware

## 7. Roadmap

## Phase 0 — Foundation and Product Definition

- Repo scaffold and tracking workflow
- Lock product assumptions, safety model, and evaluation criteria
- Define the first bounded action vocabulary
- Choose the first end-to-end showcase tasks
- Establish typed runtime contracts for observations, intents, proposals, decisions, and results

## Phase 1 — Contracts and Command Broker

- Define multimodal input packets and normalized intent objects
- Define action proposal and action result schemas
- Build a broker that maps intents to bounded OS actions
- Implement confirmation policy by action risk

## Phase 2 — Voice MVP

- Realtime audio capture and transcription
- Turn segmentation and interruption
- Voice-only intent parsing
- Spoken and visual acknowledgement
- Push-to-talk default interaction mode with room for a later hybrid model

## Phase 3 — Gaze MVP

- Calibration flow
- Gaze target inference
- Dwell and eye-gesture actions
- Tolerance models for noisy tracking
- Large-target strategy suitable for webcam-only tracking

## Phase 4 — Multimodal Fusion

- Fuse gaze target with spoken deictic commands ("this", "that", "here")
- Add ambiguity handling, confidence thresholds, and recovery flows
- Add task-level evaluation for speed, error rate, and accidental actions
- Improve target disambiguation using desktop context and recent history

## Phase 5 — Productization

- Desktop overlay or shell
- Session logging and replay
- Configuration, calibration persistence, and hardware profiles
- Packaging for repeatable local use
- Cross-platform abstraction hardening

## Phase 6 — MVP Readiness

- Add a real macOS microphone-to-transcript path
- Define readiness targets for latency, accuracy, and accidental actions
- Expose a tester-facing live command harness
- Tighten permission and failure handling for the first real user trial

## Phase 7 — Webcam MVP Completion

- Add a real tester-facing live webcam calibration path
- Persist live webcam calibration profiles for repeated runs
- Expose a live webcam gaze command path with structured recovery behavior
- Update MVP readiness so webcam gaze stands beside the live voice path

## Phase 8 — Live Multimodal Demo

- Add a real tester-facing `fusion-live` path that reuses saved webcam calibration and live macOS speech capture
- Keep the first live multimodal sequence simple: short gaze grounding first, then voice, then optional live confirmation
- Preserve the deterministic scripted `fusion-smoke` path alongside the real live demo path
- Tighten structured recovery behavior for missing calibration, missing gaze lock, camera denial, and speech denial

## 8. Remaining Open Questions

These are the important questions still worth resolving before implementation gets too opinionated:

1. What are the first 10 commands and interaction flows we will optimize for?
2. Which actions can safely auto-execute in MVP, and which must always confirm?
3. Should typing be command-only at first, or also support free dictation mode?
4. Will the initial UX be always-listening, push-to-talk, or a hybrid?
5. What latency target is acceptable for the system to still feel direct?
6. What visual overlay is sufficient for gaze targeting without becoming distracting?

## 9. Delivery Workflow

Each phase should follow the same execution pattern:

1. Write the top-level phase spec plan.
2. Write the detailed subplans.
3. Review and patch the plans.
4. Implement against the current plan set.
5. Run a code-review gate at phase end.
6. Patch review findings before calling the phase complete.

This repo now has explicit high-level plans and detailed subplans for Phases 1 through 8 under `docs/plans/`.

## 10. Current Milestone Status

- **Phase 0:** complete
- **Phase 1:** complete
- **Phase 2:** complete under the reviewed push-to-talk, transcript-driven scope
- **Phase 3:** complete under the reviewed provider-agnostic gaze scope, plus the reviewed webcam realism patch
- **Phase 4:** complete with shared interaction state, deictic fusion, confidence-based clarification, fusion metrics, and a repeatable smoke harness
- **Phase 5:** complete with terminal overlay state, session logging and replay, persisted runtime settings/calibration, a local CLI entry point, and hardened platform capability seams
- **Phase 6:** complete with a real macOS live voice path, explicit readiness targets, a tester-facing smoke command, and a recorded MVP-readiness review gate
- **Phase 7:** complete with live webcam calibration, a persisted `gaze-live` tester path, webcam-specific MVP readiness updates, and a recorded review gate
- **Phase 8:** complete with a live sequential multimodal demo path, a repo-local `phase8` smoke wrapper, structured live fusion recovery payloads, and a recorded review gate
