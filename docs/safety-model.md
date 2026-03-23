# Safety Model

## Principle

The model can interpret and propose. It does not directly control the computer. A command broker owns the final execution decision.

## Broker Decisions

- `allow` - execute immediately
- `confirm` - ask the user before execution
- `clarify` - ask the user to disambiguate
- `reject` - do not execute

## Risk Levels

### L0 - No side effects

Examples:

- highlight a target
- show likely options
- repeat transcript or explain the planned action

Default policy: `allow`

### L1 - Reversible navigation

Examples:

- small scroll
- focus a window
- open app switcher
- press `escape`

Default policy: usually `allow`

### L2 - State-changing local actions

Examples:

- click a target
- type text
- press `enter`
- close a window

Default policy in MVP: usually `confirm`

### L3 - Sensitive or destructive actions

Examples:

- send a message or email
- submit a form
- delete data
- make a purchase
- run shell or terminal commands
- enter passwords or security-sensitive values

Default policy in MVP: `reject` or require strong confirmation outside the first scope

## Hard Rules

- Never allow unrestricted shell execution from model output.
- Never auto-execute a high-risk action because confidence alone is high.
- Never rely on raw screen coordinates without a grounded target abstraction when a semantic target is available.
- Always log the proposed action, broker decision, and execution result.

## Confirmation Rules For MVP

- Clicking should usually confirm unless the product later learns a clearly safe context.
- Typing free-form text should confirm.
- Navigation keys can usually auto-execute.
- Sensitive verbs stay blocked even if the model proposes them correctly.

See [command-vocabulary.md](command-vocabulary.md) for the first explicit command set and default broker decisions.

## Safety UX

- Use visible target highlighting before action whenever possible.
- Show the user what the system believes it is about to do.
- Make cancellation fast and reliable.
- Prefer doing nothing over doing the wrong thing.
