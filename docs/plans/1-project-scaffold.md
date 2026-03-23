# 1: Project Scaffold

**Status:** completed
**Goal:** Create the initial repository structure and tracking workflow for a multimodal computer-interaction project focused on voice and gaze input.

## Tasks
- [x] 1. Mirror the reference project's tracking workflow
  - [x] 1-1. Create `.cursor/rules/project-tracking.mdc`
  - [x] 1-2. Create `AGENT.md`
- [x] 2. Create the initial documentation set
  - [x] 2-1. Add `README.md`
  - [x] 2-2. Add `docs/todo.md`
  - [x] 2-3. Add `docs/development-plan.md`
  - [x] 2-4. Add `docs/architecture.md`
  - [x] 2-5. Add `docs/bugs.md`
  - [x] 2-6. Add `docs/changelog.md`
  - [x] 2-7. Add `docs/llm-api-guide.md`
- [x] 3. Create the minimal package baseline
  - [x] 3-1. Add `pyproject.toml`
  - [x] 3-2. Add `src/interaction/`
  - [x] 3-3. Add `tests/`
  - [x] 3-4. Add a minimal smoke test for package wiring
- [x] 4. Capture initial product assumptions and open questions

## Decisions
- The requested reference project path `../deep-agent-system` did not exist locally; the scaffold mirrors `../deep-agent-network`, which contains the matching tracking workflow and document structure.
- Keep the initial implementation surface minimal. Do not lock in speech, eye-tracking, or desktop-automation libraries before the product is sharpened.
- Treat safety as a first-class architectural concern: model output proposes actions, but a command broker owns execution policy.
- Use Python 3.11+ for the first package scaffold. Platform/runtime choices for desktop UI remain open.

## Notes
- The next useful planning step is product-definition work: user profile, first platform, and first action slice.
- The roadmap intentionally separates voice, gaze, and fusion so the first prototypes stay testable.
