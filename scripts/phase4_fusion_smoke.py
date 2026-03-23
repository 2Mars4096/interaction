"""Manual smoke harness for the Phase 4 multimodal fusion loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.contracts import EnvironmentSnapshot, GazeObservation
from interaction.runtime import MultimodalFusionLoop
from interaction.vision import NormalizedScreenTarget


def main() -> int:
    loop = MultimodalFusionLoop()
    target = NormalizedScreenTarget(
        target_id="compose",
        label="Compose button",
        role="button",
        x=0.52,
        y=0.16,
        width=0.22,
        height=0.12,
    ).to_grounded_target(confidence=0.91)

    events = []
    events.append(
        loop.update_gaze_context(
            GazeObservation(confidence=0.88, x_norm=0.56, y_norm=0.20, fixation_ms=250),
            target,
        )
    )
    events.extend(loop.run_voice_turn("open this", EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))
    events.extend(loop.run_voice_turn("yes", EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))

    payload = {
        "events": [
            {
                "phase": event.phase,
                "message": event.message,
                "transcript": event.transcript,
                "target": event.target.model_dump(mode="json") if event.target else None,
                "decision": event.decision.model_dump(mode="json") if event.decision else None,
                "result": event.result.model_dump(mode="json") if event.result else None,
            }
            for event in events
        ],
        "metrics": loop.metrics.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
