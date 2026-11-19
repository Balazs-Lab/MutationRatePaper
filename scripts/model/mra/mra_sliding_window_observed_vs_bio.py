#!/usr/bin/env python3
"""mra sliding-window observed vs modeled mutation rate analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MRA_DIR = Path(__file__).resolve().parent
if str(_MRA_DIR) not in sys.path:
    sys.path.insert(0, str(_MRA_DIR))

os.environ.setdefault("MUT_SUPPLY_MODEL_VERSION", "mra")

from sliding_window_observed_vs_bio import main  # noqa: E402

if __name__ == "__main__":
    main()
