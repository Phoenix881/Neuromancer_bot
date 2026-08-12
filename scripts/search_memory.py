#!/usr/bin/env python3
"""CLI wrapper for read-only Neuromancer message memory search."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuromancer_bot.memory_search import main


if __name__ == "__main__":
    raise SystemExit(main())
