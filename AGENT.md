# Interaction - Agent Instructions

This file mirrors the always-on rules in `.cursor/rules/project-tracking.mdc` so agents that read `AGENT.md` follow the same project tracking workflow.

## Project Tracking Documents

All tracking docs live in `docs/`. Detailed plans live in `docs/plans/`.

### Document Inventory

| File | Purpose | When to Read | When to Update |
|------|---------|--------------|----------------|
| `docs/todo.md` | High-level task list grouped by phase, links to plan files | Start of session to pick up work | After completing or discovering tasks |
| `docs/plans/N-name.md` | Detailed breakdown of a buildable unit with nested sub-tasks | When working on that plan | Check off sub-tasks as you complete them |
| `docs/changelog.md` | Log of completed work, **latest first** (descending order) | Before starting work (to avoid re-doing) | After every meaningful change — end of session at latest |
| `docs/architecture.md` | Tech stack, directory layout, conventions, patterns | Before writing any code | When adding new modules, changing patterns, or introducing dependencies |
| `docs/bugs.md` | Known issues, failed approaches, workarounds | Before debugging or proposing solutions | When discovering bugs or when an approach fails |
| `docs/development-plan.md` | Vision, roadmap, and product decisions | Start of session | Only when scope, goals, or roadmap change |
| `README.md` | User-facing project overview, quick start, feature summary | Before adding new user-facing features | When adding features, changing setup steps, or updating the roadmap |
| `docs/llm-api-guide.md` | LLM-facing interaction protocol guide — teaches callers how to propose intents, action candidates, confirmations, and multimodal context packets | Before writing prompt contracts or agent integrations | When adding/changing intent schemas, action schemas, confirmation policies, or multimodal context structures |
| `docs/reviews/*.md` | Phase-end code review findings and gate decisions | At the end of a phase or before closing review findings | When performing a review gate or patching review findings |

### Session Start Protocol

1. Read `docs/todo.md` — identify what to work on and which plan is active
2. Read the active plan file in `docs/plans/` — get detailed sub-tasks for current work
3. Read `docs/changelog.md` (last 3–5 entries) — understand recent context
4. Read `docs/architecture.md` — respect existing patterns before writing code
5. Read `docs/development-plan.md` only if the task touches roadmap-level decisions

### After Each Modification

1. **Active plan file** — check off completed sub-tasks (`[x]`), add new sub-tasks if discovered
2. **`docs/todo.md`** — mark completed high-level items `[x]`, add newly discovered tasks as `[ ]`
3. **`docs/changelog.md`** — add a new entry at the **top** (descending order: latest first):
   ```
   ## YYYY-MM-DD
   - [category] Brief description of what changed and why
   ```
   Categories: `feat`, `fix`, `refactor`, `docs`, `test`, `infra`. Never delete entries.
4. **`docs/architecture.md`** — update if you introduced new files, changed directory structure, added dependencies, or established new patterns
5. **`docs/bugs.md`** — update if you found a bug, worked around one, or tried an approach that failed (record what and why)
6. **`docs/development-plan.md`** — update only if a roadmap milestone is completed or scope changed
7. **`README.md`** — update if you added a user-facing feature, changed setup/install steps, or completed a roadmap milestone. Keep it concise.
8. **`docs/llm-api-guide.md`** — update if you added or changed intent schemas, action proposal formats, multimodal context payloads, or model-facing confirmation rules

### Phase Delivery Workflow

For any meaningful implementation phase, follow this order:

1. Create or update the **top-level phase plan** in `docs/plans/`
2. Create the **detailed subplans** for the phase before implementation begins
3. Review the plans, patch them if needed, and only then start implementation
4. Implement against the current plan files
5. End the phase with a **code review gate** captured in `docs/reviews/`
6. Patch review findings before marking the phase complete

Do not skip from idea directly to implementation for multi-step work.

### Plan Files

Plans live in `docs/plans/` with **hierarchical numbering** that mirrors the task tree:

```text
plans/
  1-name.md
  1-1-name.md
  1-2-name.md
  2-name.md
  2-1-name.md
  2-1-1-name.md
```

The number prefix encodes the hierarchy. Sub-plans are created just-in-time. Future items stay as one-liners in `todo.md` marked `(not yet planned)`.

#### Plan file format

```markdown
# 1-1: Sub-Plan Title

**Parent:** [1-name](1-name.md)
**Status:** not-started | in-progress | completed
**Goal:** One sentence describing what this plan achieves.

## Tasks
- [ ] 1. Task
  - [ ] 1-1. Sub-task
  - [ ] 1-2. Sub-task
- [ ] 2. Task

## Decisions
- (filled in during execution)

## Notes
- (filled in during execution)
```

Top-level plans omit the **Parent** field. Sub-plans link back to their parent.

#### todo.md format

High-level items grouped by phase. Each links to its plan file.

```markdown
# Todo

## Phase 0 — Phase Name
- [x] [1-name](plans/1-name.md) — brief description
- [ ] 2: Description → (not yet planned)

## Backlog (unphased)
- [ ] Task description
```

### Rules

- Keep each document concise. Bullet points over paragraphs.
- **Changelog:** Update in **descending order** — newest entries at the top. Never delete entries.
- In `docs/bugs.md`, never remove a failed approach — it prevents retry loops.
- Plan numbering is hierarchical: top-level (1, 2, 3), sub-plans (1-1, 1-2), sub-sub-plans (1-1-1). Depth expands just-in-time.
- When unsure whether to update a doc, update it. The cost of a stale doc is higher than a redundant entry.
