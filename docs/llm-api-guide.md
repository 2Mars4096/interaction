# LLM API Guide

This file defines the model-facing direction for converting multimodal context into safe computer actions.

## Current Rule

The model does not directly execute anything. It proposes a structured action candidate. The command broker owns execution.

## Product Assumptions Behind This Contract

- General computer user
- macOS-first environment
- Webcam-based gaze input
- Basic control vocabulary, not unrestricted automation

## Current Implementation Surface

The initial typed schemas now live in `interaction.contracts`. They are intentionally narrow and scaffold-grade, but they define the first shared contract layer for:

- multimodal context packets
- grounded targets
- normalized intents
- action proposals
- broker decisions
- execution results

## Contract Surface

- **Multimodal context packet:** transcript, grounded targets, active app/window, recent confirmations, session mode
- **Intent object:** normalized user goal independent of phrasing
- **Action proposal:** bounded action type plus typed arguments
- **Risk annotation:** `L0`, `L1`, `L2`, `L3`
- **Broker decision:** `allow`, `confirm`, `clarify`, `reject`
- **Action result:** `success`, `failure`, `cancelled`, `blocked`

## Draft Multimodal Context Packet

```json
{
  "transcript": {
    "text": "open this",
    "is_final": true,
    "language": "en"
  },
  "gaze": {
    "target_id": "target_12",
    "confidence": 0.81,
    "screen_region": "upper_right"
  },
  "environment": {
    "platform": "macos",
    "active_app": "Safari",
    "active_window_title": "OpenAI Docs"
  },
  "session": {
    "mode": "command",
    "last_confirmed_action": null
  },
  "grounded_targets": [
    {
      "target_id": "target_12",
      "label": "Safari tab close button",
      "role": "button",
      "confidence": 0.74
    }
  ]
}
```

## Draft Intent Shape

```json
{
  "intent": "open_target",
  "target_ref": "target_12",
  "modifiers": {},
  "confidence": 0.84,
  "needs_clarification": false
}
```

## Draft Action Proposal Shape

```json
{
  "action": "click_target",
  "arguments": {
    "target_ref": "target_12"
  },
  "confidence": 0.86,
  "risk": "L2",
  "requires_confirmation": true,
  "rationale": "User said 'open this' while gaze was stable on a clickable target.",
  "spoken_response": "I found a target under your gaze. Confirm to click it."
}
```

## Draft Allowed Actions

- `highlight_target`
- `focus_target`
- `click_target`
- `double_click_target`
- `scroll`
- `open_app`
- `switch_app`
- `press_key`
- `type_text`
- `translate_text`
- `clarify`
- `reject`

## Draft Rules

- Prefer bounded verbs such as `click_target`, `scroll`, `open_app`, `type_text`, `translate_text`.
- Reference grounded targets rather than raw screen coordinates whenever possible.
- Include a rationale that states why the target and action were chosen.
- If transcript or gaze context is ambiguous, return clarification instead of guessing.
- Never emit raw shell commands or unrestricted OS instructions.
- `L2` and `L3` actions should usually surface `requires_confirmation: true` in MVP.

## Voice Input Normalization and Recognition Evidence

- Command syntax accepts sentence punctuation, polite prefixes, and bare gaze activation commands. Dictation content is preserved separately from command normalization.
- Explicit recognition confidence below the configured threshold (default 0.6) prevents execution, including confirmation. Unknown confidence remains distinct from a measured low score.
- Cancellation remains available regardless of recognition confidence.
- A new non-confirmation command replaces the old pending action, including when the new command needs clarification.

## Live Context and Pending Actions

- Gaze observation age is measured with a monotonic clock. Supplying a worker timestamp preserves the original age; a delayed observation must not become fresh merely because it arrived later.
- Live fusion binds commands to recent stable gaze at command finalization. This does not imply speech-onset alignment.
- A pending confirmation expires after 15 seconds by default and is bound to the original application/window snapshot.
- Pointer confirmations additionally require fresh gaze, the same target revision, and proximity to the proposed point (default maximum normalized Euclidean distance: 0.08).
- Tracking loss or an intervening target change cancels the pointer action, even if gaze returns to the old region before the next speech result.
- Unknown context is represented explicitly. Live execution stops when the foreground app cannot be identified; scripted dry-run callers can still supply an unknown snapshot.
- Exact editable-field identity and semantic target bounds remain future Accessibility integration work.

## Calibration Evidence

- Live gaze must use a current profile compatible with the camera index and tracker/signal schema. Legacy or unvalidated profiles do not enable live control.
- Validation quality records held-out target errors, sample counts, raw dispersion, and dispersion after the fitted mapping. Fitting and validation targets are separate; validation does not modify coefficients.
- Error limits are coarse engineering gates. A passing profile does not convert eye-detection confidence into a measured probability of accurate targeting.
- Coordinates refer to the primary display. Multi-display semantic targeting remains outside the current contract.

## Gaze Continuity and Dwell

- A live lock reports sample count and actual stable duration. Its detector confidence is not a calibrated probability of accurate targeting.
- Missing, weak, discontinuous, or displaced evidence invalidates the lock and changes its revision. Bounded capture returns only a currently supported lock, even when an earlier target had more historical hits.
- A fired dwell target stays disarmed through cooldown and tracking loss. Sustained valid look-away is required before a complete fresh dwell can activate again.
- Camera delays advance timeouts; they never establish a fixation. The first observation after a target transition contributes zero live fixation time.

## Remaining Decisions

- Canonical grounded-target schema
- Final allowed action verb set for MVP
- Final risk taxonomy and confirmation table
- Whether early intent parsing is rule-based, model-based, or hybrid
- Whether dictation mode uses a different contract from command mode
