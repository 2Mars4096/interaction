"""Manual smoke harness for the Phase 3 gaze loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.contracts import EnvironmentSnapshot
from interaction.runtime import GazeTrackingLoop
from interaction.vision import CalibrationSample, DwellTrigger, GazeSample, NormalizedPoint, NormalizedScreenTarget


def main() -> int:
    loop = GazeTrackingLoop(dwell_trigger=DwellTrigger(dwell_ms=600))
    targets = [
        NormalizedScreenTarget(target_id="compose", label="Compose button", role="button", x=0.52, y=0.16, width=0.22, height=0.12),
        NormalizedScreenTarget(target_id="sidebar", label="Mail sidebar", role="panel", x=0.02, y=0.1, width=0.20, height=0.75),
    ]
    calibration_samples = [
        CalibrationSample(raw=NormalizedPoint(0.54, 0.20), expected=NormalizedPoint(0.55, 0.20)),
        CalibrationSample(raw=NormalizedPoint(0.18, 0.50), expected=NormalizedPoint(0.18, 0.50)),
        CalibrationSample(raw=NormalizedPoint(0.70, 0.20), expected=NormalizedPoint(0.70, 0.20)),
    ]
    trace = [
        GazeSample(point=NormalizedPoint(0.55, 0.20), confidence=0.82, delta_ms=200),
        GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.84, delta_ms=200),
        GazeSample(point=NormalizedPoint(0.56, 0.20), confidence=0.85, delta_ms=200),
        GazeSample(point=NormalizedPoint(0.57, 0.20), confidence=0.86, delta_ms=200),
    ]

    events = []
    events.extend(loop.calibrate(calibration_samples))
    events.extend(loop.run_trace(trace, targets, EnvironmentSnapshot(active_app="Mail", active_window_title="Inbox")))

    print(
        json.dumps(
            [
                {
                    "phase": event.phase,
                    "message": event.message,
                    "observation": event.observation.model_dump(mode="json") if event.observation else None,
                    "target": event.target.model_dump(mode="json") if event.target else None,
                    "proposal": event.proposal.model_dump(mode="json") if event.proposal else None,
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
