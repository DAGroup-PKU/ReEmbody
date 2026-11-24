#!/usr/bin/env python3
"""Entry point for Stage 1 (Real1 -> Sim1) three-view source reconstruction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "rssr_reconstruct" / "rssr_campaign.py"


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"Missing Stage 1 entrypoint: {TARGET}")
    return subprocess.run([sys.executable, str(TARGET), *sys.argv[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
