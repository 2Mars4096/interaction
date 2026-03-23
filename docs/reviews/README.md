# Review Gates

Each meaningful phase should end with a code review gate.

## Purpose

- catch bugs, regressions, weak assumptions, and missing tests before a phase is marked complete
- force implementation to be evaluated against the phase plan, not only against "does it run"
- preserve review findings so they can be patched instead of forgotten

## Expected Review Flow

1. Finish the planned implementation for the phase.
2. Run the relevant tests and checks.
3. Perform a code review focused on correctness, risk, regressions, and missing coverage.
4. Record findings in `docs/reviews/YYYY-MM-DD-phase-name-review.md`.
5. Patch findings.
6. Mark the phase complete only after the review gate is cleared.

## Suggested Review Format

```markdown
# YYYY-MM-DD - Phase Name Review

## Findings
- Severity: finding with file references

## Open Questions
- Question or assumption

## Gate Decision
- pass | pass-with-followups | blocked
```
