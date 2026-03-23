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

## Pending Decisions

- Canonical grounded-target schema
- Final allowed action verb set for MVP
- Final risk taxonomy and confirmation table
- Whether early intent parsing is rule-based, model-based, or hybrid
- Whether dictation mode uses a different contract from command mode
