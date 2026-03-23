# 4-3: macOS Control Slice

**Parent:** [4-phase-1-command-broker](4-phase-1-command-broker.md)
**Status:** completed
**Goal:** Implement the first bounded macOS action adapter for a small, demoable control set.

## Tasks
- [x] 1. Define the adapter boundary
  - [x] 1-1. Separate pure broker logic from macOS side effects
  - [x] 1-2. Define action-to-adapter mappings
- [x] 2. Implement low-risk actions first
  - [x] 2-1. Highlight or focus target placeholder action
  - [x] 2-2. Scroll action
  - [x] 2-3. Open or switch app action
  - [x] 2-4. Basic keypress action (`escape`, `tab`, `enter`, `back`)
- [x] 3. Implement guarded state-changing action
  - [x] 3-1. Click action behind broker confirmation rules
- [x] 4. Add tests and a minimal smoke path
  - [x] 4-1. Adapter unit tests where possible
  - [x] 4-2. Safe local smoke harness for manual macOS verification

## Decisions
- The macOS layer starts in dry-run mode by default to avoid accidental side effects while the broker and target model are still immature.
- Scroll is currently approximated with page-up/page-down key events rather than pixel-perfect scrolling.
- `highlight_target` currently uses a safe notification placeholder rather than a true overlay.
- Live `click_target` and point-based `focus_target` use the local `interaction.platform.macos_runtime` helper and currently require explicit `screen_point` coordinates.

## Notes
- Keep the adapter small and explicit. Do not add general automation or shell execution here.
- `target_ref`-only click/focus grounding is intentionally deferred to the later gaze/fusion phases.
