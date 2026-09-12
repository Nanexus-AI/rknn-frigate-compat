from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import error_envelope, success_envelope, tensor


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "frigate"


def invoke(model: Path, config: Path, probe: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "rknn_frigate_compat", "--model", str(model), "--config", str(config), "--probe", str(probe), *extra],
        cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
    )


def inputs(tmp_path: Path, name: str = "model ; placeholder.rknn") -> tuple[Path, Path]:
    directory = tmp_path / "inputs with spaces"
    directory.mkdir()
    model = directory / name
    model.write_bytes(b"not-a-real-model")
    config = directory / "config ; test.yaml"
    config.write_bytes((FIXTURES / "supported.yaml").read_bytes())
    return model, config


def test_help_is_success_without_inputs_or_probe() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, "-m", "rknn_frigate_compat", "--help"], cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--runtime-lib-dir" in result.stdout


def test_compatible_human_and_json_are_equivalent(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    probe = fake_probe(success_envelope(runtime_file))
    before = (hashlib.sha256(model.read_bytes()).hexdigest(), hashlib.sha256(config.read_bytes()).hexdigest(), hashlib.sha256(runtime_file.read_bytes()).hexdigest())
    human = invoke(model, config, probe)
    machine = invoke(model, config, probe, "--json")
    after = (hashlib.sha256(model.read_bytes()).hexdigest(), hashlib.sha256(config.read_bytes()).hexdigest(), hashlib.sha256(runtime_file.read_bytes()).hexdigest())
    assert human.returncode == machine.returncode == 0
    assert "STRUCTURALLY_COMPATIBLE" in human.stdout
    assert "does not prove semantic compatibility" in human.stdout
    assert "mechanically checked structural scope" in human.stdout
    assert "Color channel meaning" in human.stdout
    assert "    Result\n      UNKNOWN\n" in human.stdout
    data = json.loads(machine.stdout)
    assert data["overall"] == "STRUCTURALLY_COMPATIBLE"
    assert data["layers"]["runtime_model"] == "PASS"
    assert data["runtime"]["loaded_library_path"] == str(runtime_file.resolve())
    assert data["runtime"]["api_version"] == "2.3.2"
    assert data["schema_version"] == 1
    assert data["model_path"] == str(model)
    assert "model_path" not in data["model_metadata"]
    assert data["model_metadata"]["input_count"] == 1
    assert data["model_metadata"]["inputs"][0]["dims"] == [1, 640, 640, 3]
    assert data["model_metadata"]["outputs"][0]["format"] == "NCHW"
    assert any(item["status"] == "UNKNOWN" for item in data["findings"])
    assert "Selection / Environment\n" in human.stdout
    assert "Detector\n  selected: rknn\n" in human.stdout
    assert "Handler\n  selected: yolo-generic\n" in human.stdout
    assert "  evaluated contract: yolo-generic:multipart\n" in human.stdout
    assert f"  inspected path: {model}\n" in human.stdout
    assert "  configured path: models/test.rknn\n" in human.stdout
    assert f"  loaded library path: {runtime_file.resolve()}\n" in human.stdout
    assert "  API version: 2.3.2\n" in human.stdout
    assert "Runtime / Model\n" not in human.stdout
    assert (
        "RKNN Model Metadata\n"
        "Inputs: 1\n"
        "Input 0\n"
        "  shape: [1, 640, 640, 3]\n"
        "  layout: NHWC\n"
        "  dtype: INT8\n"
        "  quantization: AFFINE\n"
        "Outputs: 2\n"
    ) in human.stdout
    assert "  status: PASS\n" in human.stdout
    assert before == after


def test_known_nine_output_and_reshape_mismatch(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    outputs = [tensor(i, [1, 64, 80, 80], "NCHW") for i in range(9)]
    probe = fake_probe(success_envelope(runtime_file, outputs=outputs))
    human = invoke(model, config, probe)
    result = invoke(model, config, probe, "--json")
    data = json.loads(result.stdout)
    assert human.returncode == result.returncode == 1
    assert data["overall"] == "STRUCTURALLY_INCOMPATIBLE"
    findings = {item["check_id"]: item for item in data["findings"]}
    assert findings["output.index_range"]["status"] == "FAIL"
    assert findings["output.0.reshape"]["expected"] == 1632000
    assert findings["output.0.reshape"]["actual"] == 409600
    assert "Detector\n  selected: rknn\n" in human.stdout
    assert "Handler\n  selected: yolo-generic\n" in human.stdout
    assert "  evaluated contract: yolo-generic:multipart\n" in human.stdout
    assert "  Output index range\n" in human.stdout
    assert "      supported indices: indices 0..2\n" in human.stdout
    assert "      output indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]\n" in human.stdout
    assert "      output count: 9\n" in human.stdout
    assert "      FAIL\n      Multipart output indices must fit the handler maps\n" in human.stdout
    assert "STRUCTURALLY_INCOMPATIBLE (exit 1)" in human.stdout


@pytest.mark.parametrize("stage", ["model_open", "rknn_init", "query_sdk_version", "query_io_count", "query_input_attr", "query_output_attr"])
def test_runtime_model_error_stages(stage: str, fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    value = error_envelope(stage, runtime_file)
    if stage in {"query_input_attr", "query_output_attr"}:
        value["error"]["tensor_index"] = 0  # type: ignore[index]
    probe = fake_probe(value, exit_code=1, stderr=b"diagnostic")
    result = invoke(model, config, probe, "--json")
    assert result.returncode == 2
    data = json.loads(result.stdout)
    assert data["overall"] == "RUNTIME_MODEL_ERROR"
    assert data["model_metadata"] is None


def test_runtime_model_error_human_preserves_identity_and_diagnostic(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    probe = fake_probe(error_envelope("rknn_init", runtime_file), exit_code=1, stderr=b"diagnostic")
    result = invoke(model, config, probe)
    assert result.returncode == 2
    assert "Selection / Environment\n" in result.stdout
    assert "Detector\n  selected: rknn\n" in result.stdout
    assert "Handler\n  selected: yolo-generic\n" in result.stdout
    assert f"  inspected path: {model}\n" in result.stdout
    assert f"  loaded library path: {runtime_file.resolve()}\n" in result.stdout
    assert "RKNN Model Metadata\n  unavailable\n" in result.stdout
    assert "  status: FAIL\n" in result.stdout
    assert "FAIL operation.failure: probe rknn_init: SYNTHETIC: synthetic failure" in result.stdout
    assert "RUNTIME_MODEL_ERROR (exit 2)" in result.stdout


@pytest.mark.parametrize("stage", ["startup", "runtime_identity", "destroy", "serialization"])
def test_internal_error_stages(stage: str, fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    probe = fake_probe(error_envelope(stage, runtime_file), exit_code=1)
    result = invoke(model, config, probe, "--json")
    assert result.returncode == 5
    data = json.loads(result.stdout)
    assert data["overall"] == "INTERNAL_ERROR"
    assert data["model_metadata"] is None


def test_primary_failure_preserves_primary_mapping(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    value = error_envelope("rknn_init", runtime_file)
    value["cleanup"] = {"stage": "destroy", "native_code": -2}
    probe = fake_probe(value, exit_code=1, stderr=b"cleanup also failed")
    result = invoke(model, config, probe, "--json")
    assert result.returncode == 2


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("unsupported-handler.yaml", "unsupported handler"), ("preset.yaml", "unsupported normalization"), ("plus.yaml", "unsupported normalization")],
)
def test_unsupported_contracts_do_not_run_handler_rules(fixture: str, expected: str, fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model = tmp_path / "placeholder.rknn"
    model.write_bytes(b"x")
    probe = fake_probe(success_envelope(runtime_file))
    result = invoke(model, FIXTURES / fixture, probe, "--json")
    data = json.loads(result.stdout)
    assert result.returncode == 4
    assert any(expected in item["summary"] for item in data["findings"])
    assert not any(item["check_id"].startswith("output.") for item in data["findings"])
    assert data["model_metadata"]["input_count"] == 1
    assert data["model_metadata"]["output_count"] == 2


def test_unsupported_human_preserves_selected_contract_and_diagnostic(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model = tmp_path / "placeholder.rknn"
    model.write_bytes(b"x")
    probe = fake_probe(success_envelope(runtime_file))
    result = invoke(model, FIXTURES / "unsupported-handler.yaml", probe)
    assert result.returncode == 4
    assert "Selection / Environment\n" in result.stdout
    assert "Detector\n  selected: rknn\n" in result.stdout
    assert "Handler\n  selected: yolox\n" in result.stdout
    assert "evaluated contract" not in result.stdout
    assert "RKNN Model Metadata\nInputs: 1\n" in result.stdout
    assert "Input 0\n  shape: [1, 640, 640, 3]\n" in result.stdout
    assert "Outputs: 2\n" in result.stdout
    assert "FAIL operation.failure: unsupported handler: yolox" in result.stdout
    assert "INDETERMINATE_UNSUPPORTED (exit 4)" in result.stdout


def test_single_output_is_unsupported(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    probe = fake_probe(success_envelope(runtime_file, outputs=[tensor(0, [1, 84, 8400], "UNKNOWN")]))
    result = invoke(model, config, probe, "--json")
    assert result.returncode == 4


def test_invalid_input_and_malformed_protocol(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    probe = fake_probe(b"{")
    missing = invoke(tmp_path / "missing.rknn", FIXTURES / "supported.yaml", probe, "--json")
    assert missing.returncode == 3
    assert json.loads(missing.stdout)["model_metadata"] is None
    model = tmp_path / "model.rknn"
    model.write_bytes(b"x")
    malformed = invoke(model, FIXTURES / "supported.yaml", probe, "--json")
    assert malformed.returncode == 5
    assert json.loads(malformed.stdout)["model_metadata"] is None


def test_explicit_runtime_match_and_mismatch(fake_probe, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    directory = tmp_path / "model zoo runtime"
    directory.mkdir()
    loader = directory / "librknnrt.so"
    loader.write_bytes(b"selected-runtime")
    matched = fake_probe(success_envelope(loader), name="matched.py")
    ok_human = invoke(model, config, matched, "--runtime-lib-dir", str(directory))
    ok = invoke(model, config, matched, "--runtime-lib-dir", str(directory), "--json")
    ok_data = json.loads(ok.stdout)
    assert ok_human.returncode == ok.returncode == 0
    assert ok_data["runtime"]["identity_verification"] == "PASS"
    assert "  selection mode: explicit\n" in ok_human.stdout
    assert f"  requested directory: {directory.resolve()}\n" in ok_human.stdout
    assert f"  expected loader target: {loader.resolve()}\n" in ok_human.stdout
    assert "  identity verification: PASS\n" in ok_human.stdout
    assert f"  loaded library path: {loader.resolve()}\n" in ok_human.stdout
    assert "  API version: 2.3.2\n" in ok_human.stdout
    assert "  driver version: 0.9.8\n" in ok_human.stdout
    other = tmp_path / "other-runtime.so"
    other.write_bytes(b"other")
    mismatched = fake_probe(success_envelope(other), name="mismatch.py")
    bad = invoke(model, config, mismatched, "--runtime-lib-dir", str(directory), "--json")
    bad_data = json.loads(bad.stdout)
    assert bad.returncode == 2
    assert bad_data["error_subtype"] == "RUNTIME_SELECTION_ERROR"
    assert bad_data["runtime"]["identity_verification"] == "FAIL"
    assert bad_data["model_metadata"] is None
    wrong_hash = success_envelope(loader)
    wrong_hash["runtime"]["loaded_library_sha256"] = "0" * 64  # type: ignore[index]
    hash_probe = fake_probe(wrong_hash, name="hash-mismatch.py")
    hashed = invoke(model, config, hash_probe, "--runtime-lib-dir", str(directory), "--json")
    assert hashed.returncode == 2
    assert json.loads(hashed.stdout)["error_subtype"] == "RUNTIME_SELECTION_ERROR"


def test_invalid_runtime_directory_prevents_probe_launch(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    marker = tmp_path / "not-created"
    probe = fake_probe(success_envelope(runtime_file))
    result = invoke(model, config, probe, "--runtime-lib-dir", str(marker), "--json")
    assert result.returncode == 3


def test_missing_probe_is_internal(tmp_path: Path) -> None:
    model, config = inputs(tmp_path)
    result = invoke(model, config, tmp_path / "missing-probe", "--json")
    assert result.returncode == 5


def test_model_symlink_and_directory_validation(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    target, config = inputs(tmp_path)
    link = tmp_path / "linked model.rknn"
    link.symlink_to(target)
    probe = fake_probe(success_envelope(runtime_file))
    assert invoke(link, config, probe, "--json").returncode == 0
    assert invoke(tmp_path, config, probe, "--json").returncode == 3


def test_python_probe_path_never_enables_shell() -> None:
    source = (ROOT / "src/rknn_frigate_compat/probe.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "shell=False" in source
