"""Live webcam smoke harness for the Phase 3 OpenCV gaze provider patch."""

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
from interaction.runtime import GazeTrackingLoop
from interaction.vision import DwellTrigger, NormalizedScreenTarget, OpenCVWebcamGazeProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live webcam-backed Phase 3 gaze smoke scenario.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--dwell-ms", type=int, default=700)
    args = parser.parse_args()

    provider = OpenCVWebcamGazeProvider(camera_index=args.camera_index)
    loop = GazeTrackingLoop(dwell_trigger=DwellTrigger(dwell_ms=args.dwell_ms))
    targets = [
        NormalizedScreenTarget(target_id="left_panel", label="Left panel", role="panel", x=0.0, y=0.0, width=0.33, height=1.0),
        NormalizedScreenTarget(target_id="center_panel", label="Center panel", role="panel", x=0.33, y=0.0, width=0.34, height=1.0),
        NormalizedScreenTarget(target_id="right_panel", label="Right panel", role="panel", x=0.67, y=0.0, width=0.33, height=1.0),
    ]

    events = []
    try:
        provider.open()
        for _ in range(args.frames):
            reading = provider.read(delta_ms=100)
            if reading is None:
                events.append({"phase": "recovering", "message": "No coarse gaze reading from webcam provider."})
                continue
            sample_events = loop.process_sample(
                reading.sample,
                targets,
                EnvironmentSnapshot(active_app="Webcam Smoke Harness", active_window_title="Live Camera"),
            )
            events.extend(
                {
                    "phase": event.phase,
                    "message": event.message,
                    "observation": event.observation.model_dump(mode="json") if event.observation else None,
                    "target": event.target.model_dump(mode="json") if event.target else None,
                    "proposal": event.proposal.model_dump(mode="json") if event.proposal else None,
                    "result": event.result.model_dump(mode="json") if event.result else None,
                    "provider": {
                        "frame_size": list(reading.frame_size),
                        "face_bbox": list(reading.face_bbox),
                        "eye_boxes": [list(box) for box in reading.eye_boxes],
                    },
                }
                for event in sample_events
            )
    except RuntimeError as exc:
        events.append(
            {
                "phase": "recovering",
                "message": str(exc),
                "note": "If camera access is denied on macOS, grant permission to the current Python process and rerun.",
            }
        )
    finally:
        provider.close()

    print(json.dumps(events, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
