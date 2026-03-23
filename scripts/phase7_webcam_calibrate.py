"""Manual wrapper for the Phase 7 webcam calibration flow."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.app import main


if __name__ == "__main__":
    argv = ["gaze-calibrate", *sys.argv[1:]]
    raise SystemExit(main(argv))
