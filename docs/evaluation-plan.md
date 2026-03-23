# Evaluation Plan

## Goal

Decide whether the webcam-first macOS prototype is good enough to keep building.

## Evaluation Themes

- latency
- task success
- accidental actions
- gaze usability
- trust and recoverability

## Initial Showcase Tasks

- open a looked-at item
- click a highlighted target after a voice command
- scroll a page by voice
- switch to a named app
- press a simple navigation key
- dictate a short phrase into the current field
- translate a selected or dictated phrase

## Metrics

### Latency

- time to listening indicator
- time to first partial transcript
- time to final action proposal
- time from final command to execution or confirmation prompt

### Task Success

- completion rate on each showcase task
- retries needed per task
- clarification rate

### Accidental Actions

- false execution rate
- wrong-target execution rate
- cancellation-before-completion rate

### Gaze Quality

- stable-target lock rate on large targets
- recalibration frequency
- dropouts during normal seated use
- false highlight or false trigger rate during dwell-based selection

### User Trust

- whether users can explain what the system thought they meant
- whether users feel comfortable correcting it
- whether users feel the confirmation rate is too high or too low

## Draft Acceptance Targets

These are now the first MVP tester targets:

- the live voice path should produce either a final transcript or a structured failure result within 10 seconds of starting a 4-second capture window
- the live webcam calibration path should save a usable profile with at least 3 successful calibration targets or fail structurally
- the live webcam gaze path should produce either grounded-target/recovery events or a structured camera failure within 5 seconds of startup
- confirmation or execution should follow a successful bounded voice command within roughly 1 to 3 seconds after transcript finalization
- on a 10-command manual bounded-command checklist, the first tester path should succeed or clarify on at least 8 runs
- `L2` and `L3` actions must not auto-execute without confirmation
- permission failures should always recover with a structured error payload and no traceback
- acceptable performance remains limited to large gaze targets before attempting fine-grained precision

## Method

- Start with scripted internal task runs on a fixed showcase task list.
- Record transcript, gaze target, broker decision, and outcome for each run.
- Review failures by category: transcription, target grounding, broker policy, platform execution, user confusion.
- Improve the task set and policy before expanding command vocabulary.
