# Interaction Model

## Goal

Define how voice, gaze, system state, and feedback combine into a usable control loop.

## Core Inputs

- **Voice:** expresses intent, modifiers, and confirmation
- **Gaze:** expresses likely target or focus region
- **Desktop context:** active app, active window, available grounded targets
- **Fallback input:** keyboard or mouse when recovery is needed

## Interaction States

### Idle

- System is monitoring readiness.
- Overlay is minimal.

### Listening

- Speech is being captured.
- Partial transcript may appear.

### Interpreting

- Transcript, gaze, and desktop context are fused.
- A target or ambiguity set is identified.

### Confirming

- The system pauses for explicit user approval because risk is not low enough for auto-execution.

### Executing

- The broker has allowed an action and the platform adapter is carrying it out.

### Recovering

- The system failed, lost confidence, or needs clarification.

### Calibrating

- The user is aligning webcam gaze estimation with screen regions or highlighted targets.

## Command Patterns

### Voice only

Useful when no precise target is needed.

Examples:

- "open Safari"
- "press escape"
- "scroll down"

Phase 2 uses **push-to-talk** as the default interaction style for voice-only control. A later hybrid model can layer on top of the same turn pipeline.

### Gaze only

Useful for lightweight focus or selection signals.

Examples:

- dwell to highlight a likely target
- blink or facial gesture to confirm a highlighted target

Gaze-only actions should stay conservative in the first prototype. Phase 3 uses **dwell-to-highlight** as the default trigger behavior; blink and other gestures are deferred.

### Voice plus gaze

This is the main interaction pattern.

Examples:

- "open this"
- "click that"
- "translate this into Chinese"
- "reply here"

## Grounding Rules

- Gaze should ground deictic phrases such as "this", "that", and "here".
- When gaze confidence is low, the system should highlight likely targets or ask for clarification.
- The first version should prefer large and semantically meaningful targets over tiny pixel-precise targets.
- The first gaze MVP should assume large targets and stable fixation windows rather than fine cursor precision.
- Phase 4 fusion should use a short-lived gaze context window so deictic voice commands resolve against the most recent stable target rather than stale attention.

## Feedback Requirements

- **Transcript feedback:** show what was heard
- **Target feedback:** show what object or region is currently grounded
- **Decision feedback:** show whether the action will execute, needs confirmation, or was rejected
- **Result feedback:** show success, failure, or cancellation
- **Listening feedback:** show whether push-to-talk capture is idle, listening, interpreting, confirming, or executing

## Confirmation Model

- Low-risk actions can auto-execute.
- Medium-risk actions should usually confirm in MVP.
- High-risk or sensitive actions should block or require strong confirmation.

## Recovery Paths

- "No, not that"
- "Cancel"
- "Show options"
- "Try again"
- keyboard/mouse fallback without breaking the session
