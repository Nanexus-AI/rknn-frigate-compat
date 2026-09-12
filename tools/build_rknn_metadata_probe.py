#!/usr/bin/env python3
"""Build the target-local probe from explicit RKNN development paths."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-dir", required=True, type=Path)
    parser.add_argument("--library-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not (args.include_dir / "rknn_api.h").is_file():
        parser.error("--include-dir must contain rknn_api.h")
    if not args.library_dir.is_dir():
        parser.error("--library-dir must be a directory")
    destination = args.prefix / "libexec" / "rknn-frigate-compat" / "rknn-metadata-probe"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        f"-I{args.include_dir}", f"-I{root}", str(root / "rknn_metadata_probe.c"),
        f"-L{args.library_dir}", "-lrknnrt", "-ldl", "-o", str(destination),
    ]
    subprocess.run(command, check=True, shell=False)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
