"""Stable human and JSON report renderers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .models import Finding, Layer, ModelMetadata, Report, Status, TensorMetadata, layer_results


_UNAVAILABLE = "unavailable"

_RUNTIME_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("selection_mode", "selection mode"),
    ("requested_directory", "requested directory"),
    ("expected_loader_target", "expected loader target"),
    ("expected_loader_sha256", "expected loader sha256"),
    ("loaded_library_path", "loaded library path"),
    ("loaded_library_sha256", "loaded library sha256"),
    ("identity_verification", "identity verification"),
    ("api_version", "API version"),
    ("driver_version", "driver version"),
    ("inherited_loader_environment", "inherited loader environment"),
)

_STRUCTURAL_LABELS = {
    "input.count": "Input count",
    "input.rank": "Input rank",
    "input.batch": "Input batch",
    "input.height": "Input height",
    "input.width": "Input width",
    "input.channels": "Input channels",
    "input.layout": "Input layout",
    "input.dtype": "Input dtype",
    "input.quantization": "Input quantization",
    "output.index_range": "Output index range",
    "output.scale_completeness": "Output scale completeness",
}

_SEMANTIC_LABELS = {
    "semantic.color": "Color channel meaning",
    "semantic.preprocessing": "Preprocessing / normalization",
    "semantic.dfl": "DFL representation",
    "semantic.objectness": "Objectness semantics",
    "semantic.classes": "Class-score semantics",
    "semantic.boxes": "Bounding-box encoding",
    "semantic.anchors": "Anchor semantics",
    "semantic.labels": "Label semantics",
    "semantic.dequantization": "RKNN Lite dequantization behavior",
}

_STRUCTURAL_VALUE_LABELS = {
    "input.count": ("count", "count"),
    "input.rank": ("rank", "rank"),
    "input.batch": ("batch", "batch"),
    "input.height": ("height", "height"),
    "input.width": ("width", "width"),
    "input.channels": ("channels", "channels"),
    "input.layout": ("layout", "layout"),
    "input.dtype": ("dtype", "dtype"),
    "input.quantization": ("quantization", "quantization"),
    "output.index_range": ("supported indices", "output indices"),
    "output.scale_completeness": ("required output count", "reported output count"),
}

_OUTPUT_CHECK = re.compile(r"^output\.(\d+)\.(rank|layout|reshape)$")


def render_json(report: Report) -> str:
    return json.dumps(report.data(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finding_line(finding: Finding) -> str:
    handler = f" [handler: {finding.handler}]" if finding.handler is not None else ""
    return f"  {finding.status.value} {finding.check_id}{handler}: {finding.summary}"


def _render_layer(lines: list[str], report: Report, layer: Layer) -> None:
    lines.append(f"  status: {layer_results(report.findings)[layer.value]}")
    lines.extend(
        _finding_line(item)
        for item in report.findings
        if item.layer is layer and item.status is not Status.PASS
    )


def _stable_value(value: object | None, *, nested: bool = False) -> str:
    """Format finding values deterministically without interpreting them."""
    if value is None:
        return _UNAVAILABLE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False) if nested else value
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        rendered = (
            f"{_stable_value(key, nested=True)}: {_stable_value(item, nested=True)}"
            for key, item in items
        )
        return "{" + ", ".join(rendered) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_value(item, nested=True) for item in value) + "]"
    if isinstance(value, tuple):
        suffix = "," if len(value) == 1 else ""
        return "(" + ", ".join(_stable_value(item, nested=True) for item in value) + suffix + ")"
    return str(value)


def _output_check(finding: Finding) -> tuple[int, str] | None:
    matched = _OUTPUT_CHECK.fullmatch(finding.check_id)
    if matched is None:
        return None
    return int(matched.group(1)), matched.group(2)


def _structural_title(finding: Finding) -> str:
    known = _STRUCTURAL_LABELS.get(finding.check_id)
    if known is not None:
        return known
    output = _output_check(finding)
    if output is None:
        return finding.check_id
    index, subject = output
    return f"Output {index} {subject}"


def _structural_value_labels(finding: Finding) -> tuple[str, str]:
    known = _STRUCTURAL_VALUE_LABELS.get(finding.check_id)
    if known is not None:
        return known
    output = _output_check(finding)
    if output is None:
        return "value", "value"
    _index, subject = output
    if subject == "reshape":
        return "required elements", "actual elements"
    return subject, subject


def _finding_output_tensor(finding: Finding, model: ModelMetadata | None) -> TensorMetadata | None:
    output = _output_check(finding)
    if output is None or model is None:
        return None
    index, _subject = output
    matches = [tensor for tensor in model.outputs if tensor.index == index]
    return matches[0] if len(matches) == 1 else None


def _append_comparison_values(
    lines: list[str],
    finding: Finding,
    model: ModelMetadata | None,
) -> None:
    expected_label, actual_label = _structural_value_labels(finding)
    lines.extend([
        "    Frigate expects",
        f"      {expected_label}: {_stable_value(finding.expected)}",
        "    RKNN model reports",
    ])

    output = _output_check(finding)
    tensor = _finding_output_tensor(finding, model)
    if output is not None and output[1] == "reshape" and tensor is not None:
        lines.append(f"      shape: {_shape(tensor.dims)}")
        lines.append(f"      layout: {tensor.format}")
    lines.append(f"      {actual_label}: {_stable_value(finding.actual)}")
    if finding.check_id == "output.index_range" and model is not None:
        lines.append(f"      output count: {model.output_count}")

    lines.extend([
        "    Result",
        f"      {finding.status.value}",
        f"      {finding.summary}",
    ])


def _render_structural_layer(lines: list[str], report: Report) -> None:
    layer = Layer.FRIGATE_STRUCTURAL
    lines.append(f"  status: {layer_results(report.findings)[layer.value]}")
    findings = [item for item in report.findings if item.layer is layer]
    for finding in findings:
        if finding.status is Status.NOT_EVALUATED:
            continue
        if finding.check_id == "operation.failure":
            lines.append(_finding_line(finding))
            continue
        lines.append(f"  {_structural_title(finding)}")
        _append_comparison_values(lines, finding, report.model_metadata)


def _semantic_title(finding: Finding) -> str:
    return _SEMANTIC_LABELS.get(finding.check_id, finding.check_id)


def _append_semantic_finding(lines: list[str], finding: Finding) -> None:
    lines.append(f"  {_semantic_title(finding)}")
    if finding.expected is not None or finding.actual is not None:
        if finding.expected is not None:
            lines.append(f"    expected: {_stable_value(finding.expected)}")
        if finding.actual is not None:
            lines.append(f"    actual: {_stable_value(finding.actual)}")
    lines.extend([
        "    Result",
        f"      {finding.status.value}",
        f"      {finding.summary}",
    ])


def _render_semantic_layer(lines: list[str], report: Report) -> None:
    layer = Layer.SEMANTIC_UNCERTAINTY
    lines.append(f"  status: {layer_results(report.findings)[layer.value]}")
    for finding in report.findings:
        if finding.layer is not layer or finding.status is Status.NOT_EVALUATED:
            continue
        _append_semantic_finding(lines, finding)


def _overall_notes(overall: str) -> tuple[str, ...]:
    if overall == "STRUCTURALLY_COMPATIBLE":
        return (
            "Compatible within the mechanically checked structural scope of the selected supported contract.",
            "This does not prove semantic compatibility, preprocessing correctness, label or numerical correctness, successful inference, or production Frigate integration.",
        )
    if overall == "STRUCTURALLY_INCOMPATIBLE":
        return (
            "A mechanically proven structural mismatch was found against the selected supported contract.",
        )
    if overall == "RUNTIME_MODEL_ERROR":
        return (
            "Structural and semantic compatibility were not proven because Runtime or model inspection failed.",
        )
    if overall == "INDETERMINATE_UNSUPPORTED":
        return (
            "The selected Frigate contract is outside the supported mechanical scope; no compatibility or incompatibility is concluded.",
        )
    if overall == "INVALID_INPUT":
        return (
            "Input or configuration was invalid; compatibility was not evaluated.",
        )
    if overall == "INTERNAL_ERROR":
        return (
            "An internal failure prevented a complete evaluation.",
        )
    return ()


def _render_overall(lines: list[str], report: Report) -> None:
    lines.extend(["", "Overall", f"  {report.overall.value} (exit {report.exit_code})"])
    lines.extend(f"  {note}" for note in _overall_notes(report.overall.value))


def _display(value: object | None) -> str:
    if value is None:
        return _UNAVAILABLE
    return str(value)


def _evaluated_contracts(report: Report) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    configured = report.handler
    for finding in report.findings:
        handler = finding.handler
        if handler is None or handler == configured or handler in seen:
            continue
        seen.add(handler)
        values.append(handler)
    return values


def _append_group(lines: list[str], title: str, fields: list[tuple[str, object | None]]) -> None:
    lines.append(title)
    for label, value in fields:
        lines.append(f"  {label}: {_display(value)}")


def _append_runtime_group(lines: list[str], runtime: dict[str, Any]) -> None:
    lines.append("RKNN Runtime")
    if not runtime:
        lines.append(f"  {_UNAVAILABLE}")
        return
    known = {key for key, _ in _RUNTIME_FIELD_LABELS}
    for key, label in _RUNTIME_FIELD_LABELS:
        if key not in runtime:
            continue
        value = runtime[key]
        if key == "identity_verification" and value is None:
            continue
        lines.append(f"  {label}: {_display(value)}")
    for key in sorted(set(runtime) - known):
        lines.append(f"  {key}: {_display(runtime[key])}")


def _render_selection_environment(lines: list[str], report: Report) -> None:
    lines.append("Selection / Environment")
    _append_group(lines, "Detector", [("selected", report.selected_detector)])
    handler_fields: list[tuple[str, object | None]] = [("selected", report.handler)]
    for contract in _evaluated_contracts(report):
        handler_fields.append(("evaluated contract", contract))
    _append_group(lines, "Handler", handler_fields)
    _append_group(
        lines,
        "Model",
        [
            ("configured path", report.configured_model_path),
            ("inspected path", report.model_path),
        ],
    )
    _append_runtime_group(lines, report.runtime)
    _render_layer(lines, report, Layer.RUNTIME_MODEL)


def _shape(dims: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(item) for item in dims) + "]"


def _append_tensor(lines: list[str], kind: str, tensor: TensorMetadata) -> None:
    lines.append(f"{kind} {tensor.index}")
    lines.append(f"  shape: {_shape(tensor.dims)}")
    lines.append(f"  layout: {tensor.format}")
    lines.append(f"  dtype: {tensor.type}")
    lines.append(f"  quantization: {tensor.quantization_type}")


def _render_model_metadata(lines: list[str], model: ModelMetadata | None) -> None:
    lines.extend(["", "RKNN Model Metadata"])
    if model is None:
        lines.append(f"  {_UNAVAILABLE}")
        return
    lines.append(f"Inputs: {model.input_count}")
    for tensor in sorted(model.inputs, key=lambda item: item.index):
        _append_tensor(lines, "Input", tensor)
    lines.append(f"Outputs: {model.output_count}")
    for tensor in sorted(model.outputs, key=lambda item: item.index):
        _append_tensor(lines, "Output", tensor)


def render_human(report: Report) -> str:
    lines: list[str] = []
    _render_selection_environment(lines, report)
    _render_model_metadata(lines, report.model_metadata)
    lines.extend(["", "Frigate Structural Contract"])
    _render_structural_layer(lines, report)
    lines.extend(["", "Semantic / Unproven Properties"])
    _render_semantic_layer(lines, report)
    _render_overall(lines, report)
    return "\n".join(lines) + "\n"
