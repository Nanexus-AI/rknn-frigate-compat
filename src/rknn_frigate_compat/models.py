"""Small, stable data model for checker inputs, findings, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Overall(str, Enum):
    STRUCTURALLY_COMPATIBLE = "STRUCTURALLY_COMPATIBLE"
    STRUCTURALLY_INCOMPATIBLE = "STRUCTURALLY_INCOMPATIBLE"
    RUNTIME_MODEL_ERROR = "RUNTIME_MODEL_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    INDETERMINATE_UNSUPPORTED = "INDETERMINATE_UNSUPPORTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ExitCode(IntEnum):
    STRUCTURALLY_COMPATIBLE = 0
    STRUCTURALLY_INCOMPATIBLE = 1
    RUNTIME_MODEL_ERROR = 2
    INVALID_INPUT = 3
    INDETERMINATE_UNSUPPORTED = 4
    INTERNAL_ERROR = 5


class Layer(str, Enum):
    RUNTIME_MODEL = "runtime_model"
    FRIGATE_STRUCTURAL = "frigate_structural"
    SEMANTIC_UNCERTAINTY = "semantic_uncertainty"


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    UNKNOWN = "UNKNOWN"
    NOT_EVALUATED = "NOT_EVALUATED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ErrorSubtype(str, Enum):
    RUNTIME_SELECTION_ERROR = "RUNTIME_SELECTION_ERROR"


OVERALL_EXIT = {item: ExitCode[item.name] for item in Overall}
OVERALL_PRECEDENCE = (
    Overall.INVALID_INPUT,
    Overall.INTERNAL_ERROR,
    Overall.RUNTIME_MODEL_ERROR,
    Overall.INDETERMINATE_UNSUPPORTED,
    Overall.STRUCTURALLY_INCOMPATIBLE,
    Overall.STRUCTURALLY_COMPATIBLE,
)


@dataclass(frozen=True)
class FrigateConfig:
    detector: str
    configured_model_path: str | None
    model_path_kind: str
    model_type: str
    width: int
    height: int
    input_tensor: str
    input_pixel_format: str
    input_dtype: str
    provenance: dict[str, str]


@dataclass(frozen=True)
class RuntimeIdentity:
    loaded_library_path: str
    loaded_library_sha256: str
    api_version: str | None = None
    driver_version: str | None = None


@dataclass(frozen=True)
class TensorMetadata:
    index: int
    name: str
    n_dims: int
    dims: tuple[int, ...]
    format: str
    type: str
    quantization_type: str
    zero_point: int
    scale: float
    element_count: int
    byte_size: int
    width_stride: int
    byte_size_with_stride: int
    fractional_length: int | None = None


@dataclass(frozen=True)
class ModelMetadata:
    model_path: str
    input_count: int
    output_count: int
    inputs: tuple[TensorMetadata, ...]
    outputs: tuple[TensorMetadata, ...]


@dataclass(frozen=True)
class ProbeSuccess:
    runtime: RuntimeIdentity
    model: ModelMetadata


@dataclass(frozen=True)
class ProbeError:
    stage: str
    code: str
    message: str
    native_code: int | None
    tensor_index: int | None = None
    runtime: RuntimeIdentity | None = None


@dataclass(frozen=True)
class Finding:
    check_id: str
    layer: Layer
    handler: str | None
    severity: Severity
    status: Status
    summary: str
    expected: Any = None
    actual: Any = None
    evidence: tuple[str, ...] = ()

    def data(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer"] = self.layer.value
        value["severity"] = self.severity.value
        value["status"] = self.status.value
        value["evidence"] = list(self.evidence)
        return value


@dataclass(frozen=True)
class Report:
    overall: Overall
    findings: tuple[Finding, ...] = ()
    selected_detector: str | None = None
    handler: str | None = None
    model_path: str | None = None
    configured_model_path: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    error_subtype: ErrorSubtype | None = None
    model_metadata: ModelMetadata | None = None

    @property
    def exit_code(self) -> int:
        return int(OVERALL_EXIT[self.overall])

    def data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "overall": self.overall.value,
            "exit_code": self.exit_code,
            "selected_detector": self.selected_detector,
            "handler": self.handler,
            "model_path": self.model_path,
            "configured_model_path": self.configured_model_path,
            "runtime": self.runtime,
            "error_subtype": self.error_subtype.value if self.error_subtype else None,
            "layers": layer_results(self.findings),
            "findings": [item.data() for item in self.findings],
            "model_metadata": model_metadata_data(self.model_metadata),
        }


def model_metadata_data(model: ModelMetadata | None) -> dict[str, Any] | None:
    if model is None:
        return None

    def tensors(items: tuple[TensorMetadata, ...]) -> list[dict[str, Any]]:
        return [asdict(item) for item in sorted(items, key=lambda item: item.index)]

    return {
        "input_count": model.input_count,
        "inputs": tensors(model.inputs),
        "output_count": model.output_count,
        "outputs": tensors(model.outputs),
    }


def aggregate(candidates: list[Overall]) -> Overall:
    present = set(candidates)
    return next(item for item in OVERALL_PRECEDENCE if item in present)


def layer_results(findings: tuple[Finding, ...]) -> dict[str, str]:
    order = (Status.FAIL, Status.WARN, Status.UNKNOWN, Status.PASS, Status.NOT_EVALUATED)
    result: dict[str, str] = {}
    for layer in Layer:
        statuses = {item.status for item in findings if item.layer is layer}
        result[layer.value] = next(
            (status.value for status in order if status in statuses),
            Status.NOT_EVALUATED.value,
        )
    return result
