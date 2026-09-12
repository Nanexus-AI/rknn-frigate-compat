from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from conftest import error_envelope, success_envelope
from rknn_frigate_compat.models import ProbeError, ProbeSuccess, RuntimeIdentity
from rknn_frigate_compat.probe import (
    ERROR_STAGES,
    MAX_STDOUT,
    ProbeLaunchError,
    ProtocolError,
    RuntimeDirectoryError,
    RuntimeSelectionError,
    parse_protocol,
    prepare_runtime,
    run_probe,
    runtime_context,
    verify_runtime_identity,
)


def encoded(value: object) -> bytes:
    return json.dumps(value, allow_nan=True).encode()


def test_committed_protocol_fixtures_validate() -> None:
    directory = Path(__file__).parent / "fixtures" / "probe"
    success = (directory / "success-compatible.json").read_bytes()
    error = (directory / "error-envelope.json").read_bytes()
    assert isinstance(parse_protocol(success, b"", 0), ProbeSuccess)
    assert isinstance(parse_protocol(error, b"", 1), ProbeError)


def test_valid_success_unknown_fields_and_optional_fraction(runtime_file: Path) -> None:
    value = success_envelope(runtime_file)
    value["future"] = {"nested": True}
    value["model"]["inputs"][0]["fractional_length"] = 7  # type: ignore[index]
    result = parse_protocol(encoded(value), b"", 0)
    assert isinstance(result, ProbeSuccess)
    assert result.model.inputs[0].fractional_length == 7


@pytest.mark.parametrize("stage", sorted(ERROR_STAGES))
def test_every_error_stage(stage: str, runtime_file: Path) -> None:
    result = parse_protocol(encoded(error_envelope(stage, runtime_file)), b"diagnostic", 1)
    assert isinstance(result, ProbeError)
    assert result.stage == stage


@pytest.mark.parametrize(
    ("mutator", "exit_code", "stderr"),
    [
        (lambda value: value, 1, b""),
        (lambda value: value, 2, b""),
        (lambda value: value, 0, b"noise"),
        (lambda value: value.pop("protocol_version"), 0, b""),
        (lambda value: value.__setitem__("protocol_version", 2), 0, b""),
        (lambda value: value.pop("status"), 0, b""),
        (lambda value: value.__setitem__("status", "future"), 0, b""),
        (lambda value: value["runtime"].pop("api_version"), 0, b""),
        (lambda value: value["model"].__setitem__("input_count", True), 0, b""),
        (lambda value: value["model"].__setitem__("output_count", 3), 0, b""),
        (lambda value: value["model"]["outputs"][1].__setitem__("index", 0), 0, b""),
        (lambda value: value["model"]["inputs"][0].__setitem__("scale", float("nan")), 0, b""),
    ],
)
def test_invalid_success_contract(mutator, exit_code: int, stderr: bytes, runtime_file: Path) -> None:
    value = success_envelope(runtime_file)
    mutator(value)
    with pytest.raises(ProtocolError):
        parse_protocol(encoded(value), stderr, exit_code)


@pytest.mark.parametrize(
    "stdout",
    [b"", b"{", b"{}{}", b"banner {}", b"{} suffix", b"null", b"[]"],
)
def test_malformed_or_impure_stdout(stdout: bytes) -> None:
    with pytest.raises(ProtocolError):
        parse_protocol(stdout, b"", 0)


def test_oversized_stdout_is_internal_protocol_failure() -> None:
    with pytest.raises(ProtocolError, match="exceeded"):
        parse_protocol(b" " * (MAX_STDOUT + 1), b"", 0)


def test_error_exit_contradiction_and_unknown_stage(runtime_file: Path) -> None:
    value = error_envelope("rknn_init", runtime_file)
    with pytest.raises(ProtocolError):
        parse_protocol(encoded(value), b"", 0)
    value["error"]["stage"] = "anything"  # type: ignore[index]
    with pytest.raises(ProtocolError):
        parse_protocol(encoded(value), b"", 1)
    value = error_envelope("rknn_init", runtime_file)
    value["error"].pop("native_code")  # type: ignore[union-attr]
    with pytest.raises(ProtocolError):
        parse_protocol(encoded(value), b"", 1)
    value = error_envelope("rknn_init", runtime_file)
    value["error"]["code"] = ""  # type: ignore[index]
    with pytest.raises(ProtocolError):
        parse_protocol(encoded(value), b"", 1)


def test_explicit_runtime_match_and_symlinks(tmp_path: Path) -> None:
    real_dir = tmp_path / "runtime real"
    real_dir.mkdir()
    versioned = real_dir / "librknnrt.so.2"
    versioned.write_bytes(b"runtime")
    (real_dir / "librknnrt.so").symlink_to(versioned.name)
    requested = tmp_path / "runtime link"
    requested.symlink_to(real_dir, target_is_directory=True)
    selection = prepare_runtime(requested)
    identity = RuntimeIdentity(str(versioned.resolve()), selection.expected_sha256 or "")
    assert verify_runtime_identity(identity, selection) == versioned.resolve()
    assert runtime_context(selection, identity, verified=True)["identity_verification"] == "PASS"


def test_explicit_runtime_rejects_escape(tmp_path: Path) -> None:
    directory = tmp_path / "runtime"
    directory.mkdir()
    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    (directory / "librknnrt.so").symlink_to(outside)
    with pytest.raises(RuntimeDirectoryError, match="escapes"):
        prepare_runtime(directory)


@pytest.mark.parametrize("case", ["missing", "file", "missing_loader", "loader_directory"])
def test_invalid_runtime_inputs(tmp_path: Path, case: str) -> None:
    directory = tmp_path / "runtime"
    if case == "file":
        directory.write_bytes(b"x")
    elif case != "missing":
        directory.mkdir()
        if case == "loader_directory":
            (directory / "librknnrt.so").mkdir()
    with pytest.raises(RuntimeDirectoryError):
        prepare_runtime(directory)


def test_runtime_path_and_hash_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "runtime"
    directory.mkdir()
    expected = directory / "librknnrt.so"
    expected.write_bytes(b"expected")
    selection = prepare_runtime(directory)
    other = directory / "other.so"
    other.write_bytes(b"other")
    with pytest.raises(RuntimeSelectionError):
        verify_runtime_identity(RuntimeIdentity(str(other.resolve()), __import__("hashlib").sha256(b"other").hexdigest()), selection)
    with pytest.raises(RuntimeSelectionError):
        verify_runtime_identity(RuntimeIdentity(str(expected.resolve()), "0" * 64), selection)


def test_default_runtime_identity_reporting(runtime_file: Path) -> None:
    selection = prepare_runtime(None)
    digest = __import__("hashlib").sha256(runtime_file.read_bytes()).hexdigest()
    identity = RuntimeIdentity(str(runtime_file.resolve()), digest, "2.3.2", "driver")
    verify_runtime_identity(identity, selection)
    context = runtime_context(selection, identity, verified=True)
    assert context["selection_mode"] == "default"
    assert "requested_directory" not in context


def test_fake_probe_execution_and_parent_environment(fake_probe, runtime_file: Path, monkeypatch) -> None:
    value = success_envelope(runtime_file)
    probe = fake_probe(value, name="probe ; metachar.py")
    model = runtime_file.parent / "model ; name.rknn"
    model.write_bytes(b"placeholder")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/original")
    before = os.environ["LD_LIBRARY_PATH"]
    result = run_probe(probe, model, prepare_runtime(None))
    assert isinstance(result, ProbeSuccess)
    assert os.environ["LD_LIBRARY_PATH"] == before


def test_fake_probe_timeout(fake_probe, runtime_file: Path) -> None:
    probe = fake_probe(success_envelope(runtime_file), delay=0.2)
    with pytest.raises(ProbeLaunchError, match="timed out"):
        run_probe(probe, runtime_file, prepare_runtime(None), timeout=0.01)


def test_missing_and_exec_format_probe(runtime_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ProbeLaunchError):
        run_probe(tmp_path / "missing", runtime_file, prepare_runtime(None))
    bad = tmp_path / "bad"
    bad.write_text("not executable format", encoding="utf-8")
    bad.chmod(0o755)
    with pytest.raises(ProbeLaunchError):
        run_probe(bad, runtime_file, prepare_runtime(None))


def test_permission_signal_and_bounded_capture(fake_probe, runtime_file: Path, tmp_path: Path) -> None:
    denied = tmp_path / "denied"
    denied.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    denied.chmod(0o644)
    with pytest.raises(ProbeLaunchError):
        run_probe(denied, runtime_file, prepare_runtime(None))
    signaled = fake_probe(b"", signal_number=15)
    with pytest.raises(ProtocolError):
        run_probe(signaled, runtime_file, prepare_runtime(None))
    oversized = fake_probe(b"x" * (MAX_STDOUT + 65536))
    with pytest.raises(ProtocolError, match="stdout exceeded"):
        run_probe(oversized, runtime_file, prepare_runtime(None))


def test_explicit_environment_is_private_and_probe_runs_once(fake_probe, tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "runtime directory"
    directory.mkdir()
    loader = directory / "librknnrt.so"
    loader.write_bytes(b"runtime")
    selection = prepare_runtime(directory)
    counter = tmp_path / "count.txt"
    probe = fake_probe(success_envelope(loader), required_ld_prefix=str(directory.resolve()), counter_path=counter)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/parent-value")
    result = run_probe(probe, loader, selection)
    assert isinstance(result, ProbeSuccess)
    assert counter.read_text(encoding="utf-8") == "1\n"
    assert os.environ["LD_LIBRARY_PATH"] == "/parent-value"
