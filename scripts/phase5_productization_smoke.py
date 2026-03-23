"""Manual smoke wrapper for the Phase 5 productization CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction.app import main


if __name__ == "__main__":
    argv = sys.argv[1:] if len(sys.argv) > 1 else ["fusion-smoke"]
    raise SystemExit(main(argv))
