"""Manual smoke harness for the Phase 1 broker and macOS adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.contracts import ActionName, ActionProposal, EnvironmentSnapshot, RiskLevel
from interaction.control import CommandBroker
from interaction.platform import MacOSPlatformAdapter


def build_proposal(args: argparse.Namespace) -> ActionProposal:
    action = ActionName(args.action)
    arguments: dict[str, object] = {}
    if args.app_name:
        arguments["app_name"] = args.app_name
    if args.key:
        arguments["key"] = args.key
    if args.direction:
        arguments["direction"] = args.direction
    if args.text:
        arguments["text"] = args.text
    if args.target_label:
        arguments["target_label"] = args.target_label
    if args.target_ref:
        arguments["target_ref"] = args.target_ref
    if args.x is not None and args.y is not None:
        arguments["screen_point"] = {"x": args.x, "y": args.y}

    return ActionProposal(
        action=action,
        arguments=arguments,
        confidence=args.confidence,
        risk=RiskLevel(args.risk),
        requires_confirmation=args.requires_confirmation,
        rationale=args.rationale,
        spoken_response=args.spoken_response,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a manual Phase 1 macOS smoke scenario.")
    parser.add_argument(
        "--action",
        default="open_app",
        choices=[action.value for action in ActionName],
        help="Action to test.",
    )
    parser.add_argument("--risk", default="L1", choices=["L0", "L1", "L2", "L3"])
    parser.add_argument("--confidence", type=float, default=0.9)
    parser.add_argument("--requires-confirmation", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="Promote confirm decisions to allow before execution.")
    parser.add_argument("--live", action="store_true", help="Execute against macOS instead of dry-run mode.")
    parser.add_argument("--app-name")
    parser.add_argument("--key")
    parser.add_argument("--direction")
    parser.add_argument("--text")
    parser.add_argument("--target-label")
    parser.add_argument("--target-ref")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument(
        "--rationale",
        default="Manual Phase 1 smoke test.",
        help="Rationale string for the proposal.",
    )
    parser.add_argument("--spoken-response")
    args = parser.parse_args()

    proposal = build_proposal(args)
    broker = CommandBroker()
    decision = broker.decide(proposal)
    if args.confirm and decision.decision.value == "confirm":
        decision = broker.confirm(decision)

    output: dict[str, object] = {
        "proposal": proposal.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
    }

    if decision.decision.value == "allow":
        adapter = MacOSPlatformAdapter(dry_run=not args.live)
        result = adapter.execute(
            broker.build_execution_request(
                decision,
                EnvironmentSnapshot(active_app="Manual Smoke Harness"),
            )
        )
        output["result"] = result.model_dump(mode="json")

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
