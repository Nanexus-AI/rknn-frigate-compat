from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


def tensor(index: int, dims: list[int], fmt: str, *, elements: int | None = None) -> dict[str, object]:
    count = 1
    for value in dims:
        count *= value
    count = count if elements is None else elements
    return {
        "index": index,
        "name": f"tensor{index}",
        "n_dims": len(dims),
        "dims": dims,
        "format": fmt,
        "type": "INT8",
        "quantization_type": "AFFINE",
        "zero_point": -1,
        "scale": 0.1,
        "element_count": count,
        "byte_size": count,
        "width_stride": dims[-2] if len(dims) >= 2 else 0,
        "byte_size_with_stride": count,
    }


def success_envelope(
    runtime_path: Path,
    *,
    inputs: list[dict[str, object]] | None = None,
    outputs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    inputs = inputs if inputs is not None else [tensor(0, [1, 640, 640, 3], "NHWC")]
    outputs = outputs if outputs is not None else [
        tensor(0, [1, 255, 80, 80], "NCHW"),
        tensor(1, [1, 255, 40, 40], "NCHW"),
    ]
    return {
        "protocol_version": 1,
        "status": "ok",
        "runtime": {
            "api_version": "2.3.2",
            "driver_version": "0.9.8",
            "loaded_library_path": str(runtime_path.resolve()),
            "loaded_library_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        },
        "model": {
            "model_path": "synthetic.rknn",
            "input_count": len(inputs),
            "output_count": len(outputs),
            "inputs": inputs,
            "outputs": outputs,
        },
    }


def error_envelope(stage: str, runtime_path: Path | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "protocol_version": 1,
        "status": "error",
        "error": {"stage": stage, "code": "SYNTHETIC", "message": "synthetic failure", "native_code": -1},
    }
    if stage not in {"startup", "runtime_identity"}:
        assert runtime_path is not None
        value["runtime"] = {
            "loaded_library_path": str(runtime_path.resolve()),
            "loaded_library_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        }
    return value


@pytest.fixture
def fake_probe(tmp_path: Path):
    created: list[Path] = []

    def factory(
        stdout: bytes | dict[str, object],
        *,
        stderr: bytes = b"",
        exit_code: int = 0,
        delay: float = 0,
        name: str = "fake probe.py",
        signal_number: int | None = None,
        required_ld_prefix: str | None = None,
        counter_path: Path | None = None,
    ) -> Path:
        path = tmp_path / f"{len(created)}-{name}"
        out = json.dumps(stdout, allow_nan=True).encode() if isinstance(stdout, dict) else stdout
        script = (
            f"#!{sys.executable}\n"
            "import base64,os,sys,time\n"
            + (f"open({str(counter_path)!r},'a',encoding='utf-8').write('1\\n')\n" if counter_path else "")
            + (f"if os.environ.get('LD_LIBRARY_PATH','').split(':')[0] != {required_ld_prefix!r}: raise SystemExit(9)\n" if required_ld_prefix else "")
            + f"time.sleep({delay!r})\n"
            + (f"os.kill(os.getpid(),{signal_number})\n" if signal_number is not None else "")
            + f"sys.stdout.buffer.write(base64.b64decode({base64.b64encode(out)!r}))\n"
            + f"sys.stderr.buffer.write(base64.b64decode({base64.b64encode(stderr)!r}))\n"
            + f"raise SystemExit({exit_code})\n"
        )
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        created.append(path)
        return path

    return factory


@pytest.fixture
def runtime_file(tmp_path: Path) -> Path:
    path = tmp_path / "default runtime.so"
    path.write_bytes(b"synthetic-runtime-v1")
    return path
