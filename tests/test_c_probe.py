from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def compiled_probe(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("c-probe")
    library = output / "librknnrt.so"
    host_shim = ROOT / "tests/host_shim"
    subprocess.run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "-I", str(host_shim), str(ROOT / "tests/c/rknn_stub.c"), "-o", str(library),
    ], check=True)
    probe = output / "rknn-metadata-probe"
    subprocess.run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
        "-DRKNN_FRIGATE_COMPAT_HOST_SHIM",
        "-I", str(host_shim), "-I", str(ROOT / "tools"),
        str(ROOT / "tools/rknn_metadata_probe.c"), "-L", str(output),
        f"-Wl,-rpath,{output}", "-lrknnrt", "-ldl", "-o", str(probe),
    ], check=True)
    return probe


def run(probe: Path, model: Path | None, failure: str | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if failure is not None:
        environment["RKNN_STUB_FAIL"] = failure
    command = [str(probe)] + ([] if model is None else [str(model)])
    return subprocess.run(command, env=environment, text=True, capture_output=True, check=False)


def test_c_probe_success_is_protocol_pure(compiled_probe: Path, tmp_path: Path) -> None:
    model = tmp_path / "placeholder model.rknn"
    model.write_bytes(b"not-real")
    result = run(compiled_probe, model)
    value = json.loads(result.stdout)
    assert result.returncode == 0 and result.stderr == ""
    assert value["protocol_version"] == 1 and value["status"] == "ok"
    assert Path(value["runtime"]["loaded_library_path"]).name == "librknnrt.so"
    assert len(value["runtime"]["loaded_library_sha256"]) == 64
    assert value["model"]["input_count"] == 1 and value["model"]["output_count"] == 2


@pytest.mark.parametrize(
    ("failure", "stage"),
    [
        ("rknn_init", "rknn_init"),
        ("query_sdk_version", "query_sdk_version"),
        ("query_io_count", "query_io_count"),
        ("query_input_attr", "query_input_attr"),
        ("query_output_attr", "query_output_attr"),
        ("destroy", "destroy"),
    ],
)
def test_c_probe_error_stages_have_no_partial_success(compiled_probe: Path, tmp_path: Path, failure: str, stage: str) -> None:
    model = tmp_path / f"{stage}.rknn"
    model.write_bytes(b"not-real")
    result = run(compiled_probe, model, failure)
    value = json.loads(result.stdout)
    assert result.returncode == 1
    assert value["status"] == "error" and value["error"]["stage"] == stage
    assert "model" not in value


def test_c_probe_startup_and_model_open(compiled_probe: Path, tmp_path: Path) -> None:
    startup = run(compiled_probe, None)
    assert json.loads(startup.stdout)["error"]["stage"] == "startup"
    missing = run(compiled_probe, tmp_path / "missing.rknn")
    assert json.loads(missing.stdout)["error"]["stage"] == "model_open"


def test_c_probe_primary_failure_wins_over_destroy(compiled_probe: Path, tmp_path: Path) -> None:
    model = tmp_path / "primary.rknn"
    model.write_bytes(b"not-real")
    value = json.loads(run(compiled_probe, model, "query_output_attr+destroy").stdout)
    assert value["error"]["stage"] == "query_output_attr"


def test_c_probe_source_has_no_inference_path() -> None:
    source = (ROOT / "tools/rknn_metadata_probe.c").read_text(encoding="utf-8")
    assert "rknn_run(" not in source
    assert "rknn_inputs_set(" not in source
