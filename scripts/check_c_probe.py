#!/usr/bin/env python3
"""Host-only C syntax and SHA-256 verification; never links a real RKNN Runtime."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
        "-DRKNN_FRIGATE_COMPAT_HOST_SHIM",
        f"-I{ROOT / 'tests/host_shim'}", f"-I{ROOT / 'tools'}",
        str(ROOT / "tools/rknn_metadata_probe.c"),
    ], check=True)
    with tempfile.TemporaryDirectory(prefix="rknn-frigate-sha-") as temporary:
        executable = Path(temporary) / "sha256-vectors"
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
            str(ROOT / "tests/c/sha256_vectors.c"), "-o", str(executable),
        ], check=True)
        subprocess.run([str(executable)], check=True)
    print("C probe syntax: PASS")
    print("SHA-256 vectors: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
