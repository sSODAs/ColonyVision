"""Convenient entry point for the ColonyVision Test50 pipeline.

Usage:
    python run.py
    python run.py --limit 1
"""
from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "scripts" / "run_test50.py"

if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
