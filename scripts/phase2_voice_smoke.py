"""Manual smoke harness for the Phase 2 push-to-talk voice loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.contracts import EnvironmentSnapshot
from interaction.runtime import VoiceCommandLoop
from interaction.platform import MacOSPlatformAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a manual Phase 2 voice-loop smoke scenario.")
    parser.add_argument("--utterance", required=True, help="Primary push-to-talk utterance.")
    parser.add_argument("--follow-up", help="Optional second utterance, useful for yes/no confirmation turns.")
    parser.add_argument("--live", action="store_true", help="Execute through the live macOS adapter instead of dry-run mode.")
    parser.add_argument("--active-app", default="Voice Smoke Harness")
    parser.add_argument("--window-title", default="Voice Smoke Harness")
    args = parser.parse_args()

    loop = VoiceCommandLoop(adapter=MacOSPlatformAdapter(dry_run=not args.live))
    environment = EnvironmentSnapshot(active_app=args.active_app, active_window_title=args.window_title)

    events = loop.run_text_turn(args.utterance, environment)
    if args.follow_up:
        events.extend(loop.run_text_turn(args.follow_up, environment))

    print(
        json.dumps(
            [
                {
                    "phase": event.phase,
                    "transcript": event.transcript,
                    "message": event.message,
                    "decision": event.decision.model_dump(mode="json") if event.decision else None,
                    "result": event.result.model_dump(mode="json") if event.result else None,
                }
                for event in events
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
