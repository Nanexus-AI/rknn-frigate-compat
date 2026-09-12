#!/usr/bin/env python3
"""Run help from the source tree without reading product inputs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "rknn_frigate_compat", "--help"],
        cwd=ROOT, env=environment, check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
