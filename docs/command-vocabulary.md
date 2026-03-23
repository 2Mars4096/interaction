# Command Vocabulary

This document captures the Phase 1 spec-review outcome for the first bounded command set.

## First 10 Showcase Commands

| Command | Example | Default Risk | Default Broker Decision | Notes |
|---|---|---|---|---|
| `highlight_target` | "show what I'm looking at" | `L0` | `allow` | no side effects |
| `focus_target` | "focus this" | `L1` | `allow` | bounded navigation |
| `scroll` | "scroll down" | `L1` | `allow` | reversible navigation |
| `open_app` | "open Safari" | `L1` | `allow` | bounded app launch |
| `switch_app` | "switch to Notes" | `L1` | `allow` | bounded app focus |
| `press_key` with safe keys | "press escape" | `L1` | `allow` | allow-list: `escape`, `tab`, `enter`, `back` |
| `click_target` | "click this" | `L2` | `confirm` | state-changing pointer action |
| `double_click_target` | "open this" | `L2` | `confirm` | same as click, higher activation risk |
| `right_click_target` | "right click this" | `L2` | `confirm` | secondary pointer action |
| `type_text` | "type hello there" | `L2` | `confirm` | local state change |
| `translate_text` | "translate this into Chinese" | `L2` | `confirm` | transforms or inserts text |

## Rejected In MVP

- unrestricted shell or terminal execution
- message sending or form submission
- purchases, file deletion, password entry, or other clearly sensitive actions

## Notes

- `L3` actions are blocked by default in the MVP broker policy.
- The first safe-key allow-list is intentionally tiny.
- Webcam gaze precision is assumed to be coarse, so target-driven commands should prefer confirmation over aggressive auto-execution.
