"""Direct, handler-scoped mechanical compatibility checks."""

from __future__ import annotations

from .models import Finding, FrigateConfig, Layer, ModelMetadata, Severity, Status, TensorMetadata


SOURCE = "Frigate 77a66e75c61862b048a07c1295877f4b31343504"


def _finding(
    check_id: str,
    status: Status,
    summary: str,
    expected: object = None,
    actual: object = None,
    *,
    handler: str | None = None,
    layer: Layer = Layer.FRIGATE_STRUCTURAL,
) -> Finding:
    severity = Severity.ERROR if status is Status.FAIL else Severity.WARNING if status in {Status.WARN, Status.UNKNOWN} else Severity.INFO
    return Finding(check_id, layer, handler, severity, status, summary, expected, actual, (SOURCE,))


def common_input_checks(config: FrigateConfig, model: ModelMetadata) -> list[Finding]:
    findings = [_finding(
        "input.count", Status.PASS if model.input_count == 1 else Status.FAIL,
        "RKNN detection path supplies exactly one input", 1, model.input_count,
    )]
    if model.input_count != 1:
        return findings
    tensor = model.inputs[0]
    rank_ok = tensor.n_dims == 4
    findings.append(_finding(
        "input.rank", Status.PASS if rank_ok else Status.FAIL,
        "Detection input must be rank four", 4, tensor.n_dims,
    ))
    if not rank_ok:
        return findings
    normalized = tensor.format.upper()
    if normalized == "NHWC":
        batch, height, width, channels = tensor.dims
    elif normalized == "NCHW":
        batch, channels, height, width = tensor.dims
    else:
        for check_id, expected in (
            ("input.batch", 1), ("input.height", config.height),
            ("input.width", config.width), ("input.channels", 3),
            ("input.layout", config.input_tensor.upper()),
        ):
            findings.append(_finding(check_id, Status.UNKNOWN, "Runtime format cannot be mapped mechanically", expected, tensor.format))
        return findings
    for check_id, expected, actual, summary in (
        ("input.batch", 1, batch, "Batch must be one"),
        ("input.height", config.height, height, "Input height must match Frigate configuration"),
        ("input.width", config.width, width, "Input width must match Frigate configuration"),
        ("input.channels", 3, channels, "Detection input must have three channels"),
        ("input.layout", config.input_tensor.upper(), normalized, "Runtime layout must match Frigate input_tensor"),
    ):
        findings.append(_finding(check_id, Status.PASS if expected == actual else Status.FAIL, summary, expected, actual))
    findings.extend([
        _finding("input.dtype", Status.UNKNOWN, "Input dtype conversion semantics are not mechanically proven", config.input_dtype, tensor.type),
        _finding("input.quantization", Status.UNKNOWN, "RKNN Lite numerical conversion/dequantization semantics are unproven", None, tensor.quantization_type),
    ])
    return findings


def _output_checks(tensor: TensorMetadata) -> list[Finding]:
    handler = "yolo-generic:multipart"
    findings = [_finding(
        f"output.{tensor.index}.rank", Status.PASS if tensor.n_dims == 4 else Status.FAIL,
        "Multipart output must be rank four", 4, tensor.n_dims, handler=handler,
    )]
    if tensor.n_dims != 4:
        return findings
    fmt = tensor.format.upper()
    layout_status = Status.PASS if fmt == "NCHW" else Status.FAIL if fmt == "NHWC" else Status.UNKNOWN
    findings.append(_finding(
        f"output.{tensor.index}.layout", layout_status,
        "Multipart handler requires channel-first output layout", "NCHW", tensor.format, handler=handler,
    ))
    if fmt != "NCHW":
        return findings
    batch, _channels, ny, nx = tensor.dims
    required = batch * 3 * 85 * ny * nx
    findings.append(_finding(
        f"output.{tensor.index}.reshape", Status.PASS if tensor.element_count == required else Status.FAIL,
        "Output must satisfy the handler reshape element equation", required, tensor.element_count, handler=handler,
    ))
    return findings


def multipart_checks(model: ModelMetadata) -> list[Finding]:
    handler = "yolo-generic:multipart"
    findings = [_finding(
        "output.index_range", Status.PASS if model.output_count <= 3 else Status.FAIL,
        "Multipart output indices must fit the handler maps", "indices 0..2", list(range(model.output_count)), handler=handler,
    )]
    for tensor in sorted(model.outputs, key=lambda item: item.index):
        findings.extend(_output_checks(tensor))
    if model.output_count == 2:
        findings.append(_finding(
            "output.scale_completeness", Status.UNKNOWN,
            "Two outputs are executable but third-scale completeness is unproven", 3, 2, handler=handler,
        ))
    return findings


def semantic_findings(config: FrigateConfig) -> list[Finding]:
    subjects = (
        ("semantic.color", "RGB/BGR semantic channel meaning"),
        ("semantic.preprocessing", "Mean/std and pixel normalization/range"),
        ("semantic.dfl", "DFL representation"),
        ("semantic.objectness", "Objectness meaning"),
        ("semantic.classes", "Class-score meaning"),
        ("semantic.boxes", "Bounding-box encoding"),
        ("semantic.anchors", "Anchor versus anchor-free meaning"),
        ("semantic.labels", "Label meaning"),
        ("semantic.dequantization", "RKNN Lite numerical/dequantization behavior"),
    )
    return [
        _finding(check_id, Status.UNKNOWN, summary + " is not exposed by structural metadata", None, None,
                 handler=config.model_type, layer=Layer.SEMANTIC_UNCERTAINTY)
        for check_id, summary in subjects
    ]
