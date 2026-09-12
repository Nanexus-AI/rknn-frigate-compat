"""Probe discovery, protocol-v1 validation, and Runtime identity proof."""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ModelMetadata, ProbeError, ProbeSuccess, RuntimeIdentity, TensorMetadata


PROTOCOL_VERSION = 1
TIMEOUT_SECONDS = 60
MAX_STDOUT = 4 * 1024 * 1024
MAX_STDERR = 64 * 1024
ERROR_STAGES = {
    "startup", "runtime_identity", "model_open", "rknn_init",
    "query_sdk_version", "query_io_count", "query_input_attr",
    "query_output_attr", "destroy", "serialization",
}
RUNTIME_ERROR_STAGES = {
    "model_open", "rknn_init", "query_sdk_version", "query_io_count",
    "query_input_attr", "query_output_attr",
}


class ProbeFailure(ValueError):
    """Base deterministic probe failure."""


class ProtocolError(ProbeFailure):
    pass


class ProbeLaunchError(ProbeFailure):
    pass


class RuntimeDirectoryError(ValueError):
    pass


class RuntimeSelectionError(ProbeFailure):
    def __init__(self, message: str, identity: RuntimeIdentity):
        super().__init__(message)
        self.identity = identity


@dataclass(frozen=True)
class RuntimeSelection:
    mode: str
    requested_directory: Path | None = None
    expected_target: Path | None = None
    expected_sha256: str | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ProtocolError(f"{name} must be a {qualifier}string")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not _is_int(value) or (minimum is not None and value < minimum):
        raise ProtocolError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _runtime(value: Any, *, success: bool) -> RuntimeIdentity:
    raw = _require_object(value, "runtime")
    path = _require_string(raw.get("loaded_library_path"), "runtime.loaded_library_path", nonempty=True)
    if not Path(path).is_absolute():
        raise ProtocolError("runtime.loaded_library_path must be absolute")
    digest = _require_string(raw.get("loaded_library_sha256"), "runtime.loaded_library_sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolError("runtime.loaded_library_sha256 must be 64 lowercase hex characters")
    api = raw.get("api_version")
    driver = raw.get("driver_version")
    if success:
        api = _require_string(api, "runtime.api_version", nonempty=True)
        driver = _require_string(driver, "runtime.driver_version")
    else:
        if api is not None:
            api = _require_string(api, "runtime.api_version")
        if driver is not None:
            driver = _require_string(driver, "runtime.driver_version")
    return RuntimeIdentity(path, digest, api, driver)


def _tensor(value: Any, location: str) -> TensorMetadata:
    raw = _require_object(value, location)
    index = _require_int(raw.get("index"), f"{location}.index", minimum=0)
    name = _require_string(raw.get("name"), f"{location}.name")
    n_dims = _require_int(raw.get("n_dims"), f"{location}.n_dims", minimum=1)
    dims = raw.get("dims")
    if not isinstance(dims, list) or len(dims) != n_dims:
        raise ProtocolError(f"{location}.dims must contain n_dims entries")
    parsed_dims = tuple(_require_int(item, f"{location}.dims", minimum=1) for item in dims)
    scale = raw.get("scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ProtocolError(f"{location}.scale must be finite")
    fractional = raw.get("fractional_length")
    if fractional is not None:
        fractional = _require_int(fractional, f"{location}.fractional_length")
    return TensorMetadata(
        index=index,
        name=name,
        n_dims=n_dims,
        dims=parsed_dims,
        format=_require_string(raw.get("format"), f"{location}.format", nonempty=True),
        type=_require_string(raw.get("type"), f"{location}.type", nonempty=True),
        quantization_type=_require_string(raw.get("quantization_type"), f"{location}.quantization_type", nonempty=True),
        zero_point=_require_int(raw.get("zero_point"), f"{location}.zero_point"),
        scale=float(scale),
        element_count=_require_int(raw.get("element_count"), f"{location}.element_count", minimum=0),
        byte_size=_require_int(raw.get("byte_size"), f"{location}.byte_size", minimum=0),
        width_stride=_require_int(raw.get("width_stride"), f"{location}.width_stride", minimum=0),
        byte_size_with_stride=_require_int(raw.get("byte_size_with_stride"), f"{location}.byte_size_with_stride", minimum=0),
        fractional_length=fractional,
    )


def _tensor_group(value: Any, count: int, name: str) -> tuple[TensorMetadata, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ProtocolError(f"model.{name} length must equal its count")
    tensors = tuple(_tensor(item, f"model.{name}[{index}]") for index, item in enumerate(value))
    if sorted(item.index for item in tensors) != list(range(count)):
        raise ProtocolError(f"model.{name} indices must be unique, contiguous, and zero-based")
    return tensors


def parse_protocol(stdout: bytes, stderr: bytes, returncode: int) -> ProbeSuccess | ProbeError:
    if len(stdout) > MAX_STDOUT or len(stderr) > MAX_STDERR:
        raise ProtocolError("probe output exceeded its limit")
    try:
        text = stdout.decode("utf-8")
        error_text = stderr.decode("utf-8")
    except UnicodeDecodeError:
        raise ProtocolError("probe output is not UTF-8") from None
    if not text.strip():
        raise ProtocolError("probe stdout is empty")
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    leading = len(text) - len(text.lstrip())
    try:
        raw, end = decoder.raw_decode(text, leading)
    except (json.JSONDecodeError, ValueError):
        raise ProtocolError("probe stdout is not one valid JSON document") from None
    if text[end:].strip():
        raise ProtocolError("probe stdout contains trailing non-whitespace data")
    envelope = _require_object(raw, "protocol envelope")
    if not _is_int(envelope.get("protocol_version")) or envelope["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported or missing protocol_version")
    status = envelope.get("status")
    if status not in {"ok", "error"}:
        raise ProtocolError("status must be ok or error")
    if status == "ok":
        if returncode != 0:
            raise ProtocolError("success envelope requires exit 0")
        if error_text:
            raise ProtocolError("success envelope requires empty stderr")
        runtime = _runtime(envelope.get("runtime"), success=True)
        model = _require_object(envelope.get("model"), "model")
        input_count = _require_int(model.get("input_count"), "model.input_count", minimum=0)
        output_count = _require_int(model.get("output_count"), "model.output_count", minimum=0)
        return ProbeSuccess(
            runtime=runtime,
            model=ModelMetadata(
                model_path=_require_string(model.get("model_path"), "model.model_path"),
                input_count=input_count,
                output_count=output_count,
                inputs=_tensor_group(model.get("inputs"), input_count, "inputs"),
                outputs=_tensor_group(model.get("outputs"), output_count, "outputs"),
            ),
        )
    if returncode != 1:
        raise ProtocolError("error envelope requires exit 1")
    error = _require_object(envelope.get("error"), "error")
    stage = error.get("stage")
    if stage not in ERROR_STAGES:
        raise ProtocolError("error.stage is unknown")
    if "native_code" not in error:
        raise ProtocolError("error.native_code is required")
    native = error["native_code"]
    if native is not None and not _is_int(native):
        raise ProtocolError("error.native_code must be an integer or null")
    tensor_index = error.get("tensor_index")
    indexed = stage in {"query_input_attr", "query_output_attr"}
    if tensor_index is not None:
        if not indexed:
            raise ProtocolError("error.tensor_index is allowed only for indexed queries")
        tensor_index = _require_int(tensor_index, "error.tensor_index", minimum=0)
    runtime = None
    if stage not in {"startup", "runtime_identity"}:
        runtime = _runtime(envelope.get("runtime"), success=False)
    elif "runtime" in envelope:
        runtime = _runtime(envelope["runtime"], success=False)
    return ProbeError(
        stage=stage,
        code=_require_string(error.get("code"), "error.code", nonempty=True),
        message=_require_string(error.get("message"), "error.message"),
        native_code=native,
        tensor_index=tensor_index,
        runtime=runtime,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def prepare_runtime(directory: Path | None) -> RuntimeSelection:
    if directory is None:
        return RuntimeSelection("default")
    try:
        canonical = directory.resolve(strict=True)
    except OSError as exc:
        raise RuntimeDirectoryError(f"invalid Runtime directory: {exc}") from None
    if not canonical.is_dir() or not os.access(canonical, os.R_OK | os.X_OK):
        raise RuntimeDirectoryError("Runtime directory must be readable and searchable")
    loader = canonical / "librknnrt.so"
    try:
        target = loader.resolve(strict=True)
    except OSError as exc:
        raise RuntimeDirectoryError(f"invalid Runtime loader: {exc}") from None
    if not target.is_file() or not os.access(target, os.R_OK):
        raise RuntimeDirectoryError("Runtime loader target must be a readable regular file")
    if not _contained(target, canonical):
        raise RuntimeDirectoryError("Runtime loader symlink escapes the requested directory")
    try:
        digest = hash_file(target)
    except OSError as exc:
        raise RuntimeDirectoryError(f"cannot hash Runtime loader: {exc}") from None
    return RuntimeSelection("explicit", canonical, target, digest)


def default_probe_path() -> Path:
    return Path(sys.prefix) / "libexec" / "rknn-frigate-compat" / "rknn-metadata-probe"


def resolve_probe(override: Path | None) -> Path:
    path = override if override is not None else default_probe_path()
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise ProbeLaunchError(f"probe is unavailable: {exc}") from None
    if not canonical.is_file() or not os.access(canonical, os.X_OK):
        raise ProbeLaunchError("probe must be an executable regular file")
    return canonical


def verify_runtime_identity(identity: RuntimeIdentity, selection: RuntimeSelection) -> Path:
    try:
        actual = Path(identity.loaded_library_path).resolve(strict=True)
        calculated = hash_file(actual)
    except OSError as exc:
        raise ProtocolError(f"cannot independently verify loaded Runtime: {exc}") from None
    if str(actual) != identity.loaded_library_path:
        error = "probe-reported loaded Runtime path is not canonical"
        if selection.mode == "explicit":
            raise RuntimeSelectionError(error, identity)
        raise ProtocolError(error)
    if calculated != identity.loaded_library_sha256:
        error = "probe and CLI loaded Runtime SHA-256 values differ"
        if selection.mode == "explicit":
            raise RuntimeSelectionError(error, identity)
        raise ProtocolError(error)
    if selection.mode == "explicit":
        assert selection.requested_directory is not None
        assert selection.expected_target is not None
        assert selection.expected_sha256 is not None
        if actual != selection.expected_target or not _contained(actual, selection.requested_directory):
            raise RuntimeSelectionError("actually loaded Runtime does not match requested loader target", identity)
        if calculated != selection.expected_sha256:
            raise RuntimeSelectionError("loaded Runtime hash does not match the prevalidated target", identity)
    return actual


def runtime_context(
    selection: RuntimeSelection,
    identity: RuntimeIdentity | None,
    *,
    verified: bool | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "selection_mode": selection.mode,
        "inherited_loader_environment": bool(os.environ.get("LD_LIBRARY_PATH")),
    }
    if selection.mode == "explicit":
        context.update({
            "requested_directory": str(selection.requested_directory),
            "expected_loader_target": str(selection.expected_target),
            "expected_loader_sha256": selection.expected_sha256,
            "identity_verification": "PASS" if verified else "FAIL" if verified is False else None,
        })
    if identity is not None:
        context.update({
            "loaded_library_path": identity.loaded_library_path,
            "loaded_library_sha256": identity.loaded_library_sha256,
            "api_version": identity.api_version,
            "driver_version": identity.driver_version,
        })
    return context


def _communicate_bounded(process: subprocess.Popen[bytes], timeout: float) -> tuple[bytes, bytes]:
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    limits = {process.stdout: MAX_STDOUT, process.stderr: MAX_STDERR}
    for stream in buffers:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    exceeded: str | None = None
    timed_out = False
    while selector.get_map() or process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0 and process.poll() is None:
            timed_out = True
            process.kill()
        for key, _ in selector.select(max(0.0, min(0.05, remaining)) if not timed_out else 0.05):
            stream = key.fileobj
            chunk = os.read(stream.fileno(), 65536)
            if not chunk:
                selector.unregister(stream)
                stream.close()
                continue
            retained = limits[stream] + 1 - len(buffers[stream])
            if retained > 0:
                buffers[stream].extend(chunk[:retained])
            if len(buffers[stream]) > limits[stream] and exceeded is None:
                exceeded = "stdout" if stream is process.stdout else "stderr"
                process.kill()
    process.wait()
    if timed_out:
        raise ProbeLaunchError("probe timed out")
    if exceeded is not None:
        raise ProtocolError(f"probe {exceeded} exceeded its limit")
    return bytes(buffers[process.stdout]), bytes(buffers[process.stderr])


def run_probe(
    probe_path: Path,
    model_path: Path,
    selection: RuntimeSelection,
    *,
    timeout: float = TIMEOUT_SECONDS,
) -> ProbeSuccess | ProbeError:
    environment = os.environ.copy()
    if selection.mode == "explicit":
        assert selection.requested_directory is not None
        old = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = str(selection.requested_directory) + ((":" + old) if old else "")
    try:
        process = subprocess.Popen(
            [str(probe_path), str(model_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            shell=False,
        )
        stdout, stderr = _communicate_bounded(process, timeout)
    except OSError as exc:
        raise ProbeLaunchError(f"cannot launch probe: {exc}") from None
    result = parse_protocol(stdout, stderr, process.returncode)
    identity = result.runtime
    if identity is not None:
        verify_runtime_identity(identity, selection)
    return result
