# 2: Product Definition Docs

**Status:** completed
**Goal:** Convert the repo from a bare scaffold into a documentation-first foundation with clear product direction, interaction rules, safety policy, and evaluation criteria.

## Tasks
- [x] 1. Lock initial product assumptions from user input
  - [x] 1-1. Capture general computer user as the primary audience
  - [x] 1-2. Capture macOS-first scope with later cross-platform intent
  - [x] 1-3. Capture webcam-only tracking for the first prototype
  - [x] 1-4. Capture a basic-controls showcase instead of a narrow niche workflow
- [x] 2. Expand core planning docs
  - [x] 2-1. Update `README.md`
  - [x] 2-2. Update `docs/development-plan.md`
  - [x] 2-3. Update `docs/architecture.md`
  - [x] 2-4. Update `docs/llm-api-guide.md`
  - [x] 2-5. Update `docs/todo.md`
- [x] 3. Add supporting product-definition docs
  - [x] 3-1. Add `docs/product-spec.md`
  - [x] 3-2. Add `docs/interaction-model.md`
  - [x] 3-3. Add `docs/safety-model.md`
  - [x] 3-4. Add `docs/evaluation-plan.md`
- [x] 4. Sync tracking memory
  - [x] 4-1. Update `docs/changelog.md`
  - [x] 4-2. Update `docs/bugs.md`

## Decisions
- The first version is explicitly for general computer users, not only an accessibility or research-only audience.
- The first platform is macOS, but architecture and roadmap should preserve a later cross-platform path.
- Webcam-only tracking is a product constraint, so early interaction design must assume coarse, noisy gaze estimates.
- The MVP should showcase a family of basic controls instead of overfitting to a single application workflow.
- Safety policy should default to conservative confirmation for state-changing actions such as clicking and typing.

## Notes
- The next implementation plan should define typed runtime contracts before choosing heavy runtime dependencies.
- The first 10 showcase commands are still worth making explicit before coding begins.
