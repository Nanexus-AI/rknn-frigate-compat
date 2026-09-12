"""Focused human-report Selection / Environment rendering tests."""

from __future__ import annotations

import pytest

from rknn_frigate_compat.models import (
    Finding,
    Layer,
    ModelMetadata,
    Overall,
    Report,
    Severity,
    Status,
    TensorMetadata,
)
from rknn_frigate_compat.report import (
    _SEMANTIC_LABELS,
    _STRUCTURAL_LABELS,
    _semantic_title,
    _stable_value,
    _structural_title,
    render_human,
)


def _finding(
    check_id: str,
    layer: Layer,
    status: Status,
    summary: str,
    *,
    handler: str | None = None,
    expected: object = None,
    actual: object = None,
) -> Finding:
    severity = (
        Severity.ERROR if status is Status.FAIL
        else Severity.WARNING if status in {Status.WARN, Status.UNKNOWN}
        else Severity.INFO
    )
    return Finding(check_id, layer, handler, severity, status, summary, expected, actual)


def _tensor(
    index: int,
    dims: list[int],
    fmt: str,
    *,
    dtype: str = "INT8",
    quantization: str = "AFFINE",
    name: str | None = None,
) -> TensorMetadata:
    count = 1
    for value in dims:
        count *= value
    return TensorMetadata(
        index=index,
        name=name if name is not None else f"tensor{index}",
        n_dims=len(dims),
        dims=tuple(dims),
        format=fmt,
        type=dtype,
        quantization_type=quantization,
        zero_point=-1,
        scale=0.1,
        element_count=count,
        byte_size=count,
        width_stride=dims[-2] if len(dims) >= 2 else 0,
        byte_size_with_stride=count,
    )


def _model(
    inputs: list[TensorMetadata],
    outputs: list[TensorMetadata],
    *,
    model_path: str = "probe-internal.rknn",
) -> ModelMetadata:
    return ModelMetadata(
        model_path,
        len(inputs),
        len(outputs),
        tuple(inputs),
        tuple(outputs),
    )


def _compatible_report(**changes) -> Report:
    findings = (
        _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "verified"),
        _finding(
            "output.index_range",
            Layer.FRIGATE_STRUCTURAL,
            Status.PASS,
            "ok",
            handler="yolo-generic:multipart",
        ),
        _finding(
            "semantic.color",
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            "not proven",
            handler="yolo-generic",
        ),
    )
    values = dict(
        overall=Overall.STRUCTURALLY_COMPATIBLE,
        findings=findings,
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/inspected/cli-model.rknn",
        configured_model_path="models/configured.rknn",
        runtime={
            "selection_mode": "default",
            "inherited_loader_environment": False,
            "loaded_library_path": "/lib/librknnrt.so",
            "loaded_library_sha256": "a" * 64,
            "api_version": "2.3.2",
            "driver_version": "0.9.8",
        },
    )
    values.update(changes)
    return Report(**values)


def test_selection_environment_success_layout_and_labels() -> None:
    text = render_human(_compatible_report())
    assert text.startswith("Selection / Environment\n")
    assert "Runtime / Model\n" not in text
    assert "Detector\n  selected: rknn\n" in text
    assert "Handler\n  selected: yolo-generic\n  evaluated contract: yolo-generic:multipart\n" in text
    assert (
        "Model\n"
        "  configured path: models/configured.rknn\n"
        "  inspected path: /inspected/cli-model.rknn\n"
    ) in text
    assert "RKNN Runtime\n  selection mode: default\n" in text
    assert "  loaded library path: /lib/librknnrt.so\n" in text
    assert "  loaded library sha256: " + ("a" * 64) + "\n" in text
    assert "  API version: 2.3.2\n" in text
    assert "  driver version: 0.9.8\n" in text
    assert "  inherited loader environment: False\n" in text
    assert "requested directory" not in text
    assert "expected loader" not in text
    assert "identity verification" not in text
    assert "  selected_detector:" not in text
    assert "  configured_model_path:" not in text
    assert "  model_path:" not in text
    assert "Frigate Structural Contract\n" in text
    assert "Semantic / Unproven Properties\n" in text
    assert "Overall\n  STRUCTURALLY_COMPATIBLE (exit 0)\n" in text
    detector_at = text.index("Detector\n")
    handler_at = text.index("Handler\n")
    model_at = text.index("Model\n")
    runtime_at = text.index("RKNN Runtime\n")
    metadata_at = text.index("RKNN Model Metadata\n")
    structural_at = text.index("Frigate Structural Contract\n")
    assert detector_at < handler_at < model_at < runtime_at < metadata_at < structural_at
    assert "RKNN Model Metadata\n  unavailable\n" in text


def test_selection_environment_explicit_runtime_identity() -> None:
    text = render_human(_compatible_report(runtime={
        "selection_mode": "explicit",
        "requested_directory": "/opt/runtime",
        "expected_loader_target": "/opt/runtime/librknnrt.so",
        "expected_loader_sha256": "b" * 64,
        "loaded_library_path": "/opt/runtime/librknnrt.so",
        "loaded_library_sha256": "b" * 64,
        "identity_verification": "PASS",
        "api_version": "2.3.2",
        "driver_version": "0.9.8",
        "inherited_loader_environment": False,
    }))
    assert "  selection mode: explicit\n" in text
    assert "  requested directory: /opt/runtime\n" in text
    assert "  expected loader target: /opt/runtime/librknnrt.so\n" in text
    assert "  expected loader sha256: " + ("b" * 64) + "\n" in text
    assert "  identity verification: PASS\n" in text
    mode_at = text.index("  selection mode: explicit\n")
    requested_at = text.index("  requested directory:")
    expected_at = text.index("  expected loader target:")
    identity_at = text.index("  identity verification: PASS\n")
    loaded_at = text.index("  loaded library path:")
    assert mode_at < requested_at < expected_at < loaded_at < identity_at


def test_selection_environment_default_runtime_omits_explicit_fields() -> None:
    text = render_human(_compatible_report())
    runtime_block = text.split("RKNN Runtime\n", 1)[1].split("\n\n", 1)[0]
    assert "selection mode: default" in runtime_block
    assert "requested_directory" not in runtime_block
    assert "requested directory" not in runtime_block
    assert "expected_loader_target" not in runtime_block
    assert "expected loader target" not in runtime_block
    assert "expected_loader_sha256" not in runtime_block
    assert "identity_verification" not in runtime_block
    assert "identity verification" not in runtime_block


def test_selection_environment_runtime_model_error_keeps_known_context() -> None:
    report = Report(
        overall=Overall.RUNTIME_MODEL_ERROR,
        findings=(
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "verified"),
            _finding("operation.failure", Layer.RUNTIME_MODEL, Status.FAIL, "probe rknn_init: SYNTHETIC: synthetic failure"),
        ),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/inspected/cli-model.rknn",
        configured_model_path="models/configured.rknn",
        runtime={
            "selection_mode": "default",
            "inherited_loader_environment": False,
            "loaded_library_path": "/lib/librknnrt.so",
            "loaded_library_sha256": "c" * 64,
            "api_version": None,
            "driver_version": None,
        },
        model_metadata=None,
    )
    text = render_human(report)
    assert "Selection / Environment\n" in text
    assert "  selected: rknn\n" in text
    assert "  selected: yolo-generic\n" in text
    assert "evaluated contract" not in text
    assert "  inspected path: /inspected/cli-model.rknn\n" in text
    assert "  loaded library path: /lib/librknnrt.so\n" in text
    assert "  API version: unavailable\n" in text
    assert "  driver version: unavailable\n" in text
    assert "FAIL operation.failure: probe rknn_init: SYNTHETIC: synthetic failure" in text
    assert "RUNTIME_MODEL_ERROR (exit 2)" in text
    assert "RKNN Model Metadata\n  unavailable\n" in text
    assert "Input 0" not in text
    assert "Output 0" not in text
    assert "Inputs:" not in text
    assert "Outputs:" not in text


def test_selection_environment_unsupported_after_inspection() -> None:
    report = Report(
        overall=Overall.INDETERMINATE_UNSUPPORTED,
        findings=(
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "verified"),
            _finding(
                "operation.failure",
                Layer.FRIGATE_STRUCTURAL,
                Status.FAIL,
                "unsupported handler: yolox",
            ),
        ),
        selected_detector="rknn",
        handler="yolox",
        model_path="/inspected/cli-model.rknn",
        configured_model_path="models/yolox.rknn",
        runtime={
            "selection_mode": "default",
            "inherited_loader_environment": False,
            "loaded_library_path": "/lib/librknnrt.so",
            "loaded_library_sha256": "d" * 64,
            "api_version": "2.3.2",
            "driver_version": "0.9.8",
        },
    )
    text = render_human(report)
    assert "Handler\n  selected: yolox\n" in text
    assert "evaluated contract" not in text
    assert "  configured path: models/yolox.rknn\n" in text
    assert "  inspected path: /inspected/cli-model.rknn\n" in text
    assert "  selection mode: default\n" in text
    assert "INDETERMINATE_UNSUPPORTED (exit 4)" in text


def test_selection_environment_missing_optional_fields_are_unavailable() -> None:
    report = Report(
        overall=Overall.INVALID_INPUT,
        findings=(
            _finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "bad config"),
        ),
    )
    text = render_human(report)
    assert "Detector\n  selected: unavailable\n" in text
    assert "Handler\n  selected: unavailable\n" in text
    assert "evaluated contract" not in text
    assert (
        "Model\n"
        "  configured path: unavailable\n"
        "  inspected path: unavailable\n"
    ) in text
    assert "RKNN Runtime\n  unavailable\n" in text
    assert "requested directory" not in text
    assert "INVALID_INPUT (exit 3)" in text


def test_selection_environment_identity_verification_fail_and_omitted_none() -> None:
    fail_text = render_human(_compatible_report(runtime={
        "selection_mode": "explicit",
        "requested_directory": "/opt/runtime",
        "expected_loader_target": "/opt/runtime/librknnrt.so",
        "expected_loader_sha256": "e" * 64,
        "loaded_library_path": "/elsewhere/librknnrt.so",
        "loaded_library_sha256": "f" * 64,
        "identity_verification": "FAIL",
        "inherited_loader_environment": True,
    }))
    assert "  identity verification: FAIL\n" in fail_text
    assert "  inherited loader environment: True\n" in fail_text

    omitted = render_human(_compatible_report(runtime={
        "selection_mode": "explicit",
        "requested_directory": "/opt/runtime",
        "expected_loader_target": "/opt/runtime/librknnrt.so",
        "expected_loader_sha256": "e" * 64,
        "identity_verification": None,
        "inherited_loader_environment": False,
    }))
    assert "identity verification" not in omitted


def test_selection_environment_does_not_use_probe_model_path() -> None:
    text = render_human(_compatible_report(
        model_path="/authoritative/cli.rknn",
        configured_model_path="models/from-config.rknn",
    ))
    assert "  inspected path: /authoritative/cli.rknn\n" in text
    assert "  configured path: models/from-config.rknn\n" in text
    assert "synthetic.rknn" not in text


def test_model_metadata_single_input_multiple_outputs() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [
            _tensor(0, [1, 255, 80, 80], "NCHW"),
            _tensor(1, [1, 255, 40, 40], "NCHW"),
        ],
    )
    text = render_human(_compatible_report(model_metadata=model))
    assert (
        "RKNN Model Metadata\n"
        "Inputs: 1\n"
        "Input 0\n"
        "  shape: [1, 640, 640, 3]\n"
        "  layout: NHWC\n"
        "  dtype: INT8\n"
        "  quantization: AFFINE\n"
        "Outputs: 2\n"
        "Output 0\n"
        "  shape: [1, 255, 80, 80]\n"
        "  layout: NCHW\n"
        "  dtype: INT8\n"
        "  quantization: AFFINE\n"
        "Output 1\n"
        "  shape: [1, 255, 40, 40]\n"
        "  layout: NCHW\n"
        "  dtype: INT8\n"
        "  quantization: AFFINE\n"
    ) in text
    assert "  dims:" not in text
    assert "  format:" not in text
    assert "  type:" not in text
    assert "  quantization_type:" not in text
    assert "zero_point" not in text
    assert "scale:" not in text
    selection_at = text.index("Selection / Environment\n")
    metadata_at = text.index("RKNN Model Metadata\n")
    structural_at = text.index("Frigate Structural Contract\n")
    assert selection_at < metadata_at < structural_at


def test_model_metadata_nine_outputs_sorted_and_compact() -> None:
    outputs = [
        _tensor(8, [1, 64, 20, 20], "NCHW"),
        _tensor(0, [1, 64, 80, 80], "NCHW"),
        _tensor(3, [1, 64, 40, 40], "NCHW"),
        _tensor(1, [1, 64, 80, 80], "NCHW"),
        _tensor(5, [1, 64, 40, 40], "NCHW"),
        _tensor(2, [1, 64, 80, 80], "NCHW"),
        _tensor(7, [1, 64, 20, 20], "NCHW"),
        _tensor(4, [1, 64, 40, 40], "NCHW"),
        _tensor(6, [1, 64, 20, 20], "NCHW"),
    ]
    text = render_human(_compatible_report(model_metadata=_model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        outputs,
    )))
    block = text.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract\n", 1)[0]
    assert block.startswith("Inputs: 1\n")
    assert "Outputs: 9\n" in block
    positions = [block.index(f"Output {index}\n") for index in range(9)]
    assert positions == sorted(positions)
    assert "YOLO" not in text
    assert "compatible" not in block.lower()
    assert "DFL" not in text
    assert "anchor" not in text.lower()


def test_model_metadata_uses_actual_tensor_indexes_not_list_position() -> None:
    model = _model(
        [_tensor(2, [1, 3, 320, 320], "NCHW", dtype="UINT8", quantization="NONE")],
        [
            _tensor(5, [1, 84, 8400], "UNKNOWN", dtype="FP16", quantization="LAYER"),
            _tensor(2, [1, 84, 8400], "UNKNOWN", dtype="FP16", quantization="LAYER"),
        ],
    )
    text = render_human(_compatible_report(model_metadata=model))
    assert "Input 2\n" in text
    assert "Input 0\n" not in text
    assert "Output 2\n" in text
    assert "Output 5\n" in text
    assert text.index("Output 2\n") < text.index("Output 5\n")
    assert "Output 0\n" not in text
    assert "Output 1\n" not in text
    assert "  dtype: UINT8\n" in text
    assert "  quantization: NONE\n" in text
    assert "  dtype: FP16\n" in text
    assert "  quantization: LAYER\n" in text


def test_model_metadata_out_of_order_inputs_and_outputs_are_sorted() -> None:
    model = _model(
        [
            _tensor(1, [1, 3, 160, 160], "NHWC"),
            _tensor(0, [1, 3, 320, 320], "NHWC"),
        ],
        [
            _tensor(3, [1, 1, 1, 1], "NCHW"),
            _tensor(1, [1, 2, 2, 2], "NCHW"),
        ],
    )
    text = render_human(_compatible_report(model_metadata=model))
    block = text.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract\n", 1)[0]
    assert block.index("Input 0\n") < block.index("Input 1\n")
    assert block.index("Output 1\n") < block.index("Output 3\n")
    assert "Inputs: 2\n" in block
    assert "Outputs: 2\n" in block


def test_model_metadata_unavailable_for_invalid_input_and_runtime_error() -> None:
    invalid = render_human(Report(
        overall=Overall.INVALID_INPUT,
        findings=(_finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "bad config"),),
    ))
    assert "RKNN Model Metadata\n  unavailable\n" in invalid
    assert "Input 0" not in invalid
    assert "Output 0" not in invalid
    assert "Inputs:" not in invalid
    assert "Outputs:" not in invalid

    runtime_error = render_human(Report(
        overall=Overall.RUNTIME_MODEL_ERROR,
        findings=(_finding("operation.failure", Layer.RUNTIME_MODEL, Status.FAIL, "probe failed"),),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/inspected/cli-model.rknn",
        model_metadata=None,
    ))
    assert "RKNN Model Metadata\n  unavailable\n" in runtime_error
    assert "Inputs:" not in runtime_error


def test_model_metadata_renders_for_unsupported_after_inspection() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(0, [1, 255, 80, 80], "NCHW"), _tensor(1, [1, 255, 40, 40], "NCHW")],
    )
    report = Report(
        overall=Overall.INDETERMINATE_UNSUPPORTED,
        findings=(
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "verified"),
            _finding(
                "operation.failure",
                Layer.FRIGATE_STRUCTURAL,
                Status.FAIL,
                "unsupported handler: yolox",
            ),
        ),
        selected_detector="rknn",
        handler="yolox",
        model_path="/inspected/cli-model.rknn",
        configured_model_path="models/yolox.rknn",
        runtime={"selection_mode": "default", "inherited_loader_environment": False},
        model_metadata=model,
    )
    text = render_human(report)
    assert "Handler\n  selected: yolox\n" in text
    assert "RKNN Model Metadata\nInputs: 1\n" in text
    assert "Input 0\n  shape: [1, 640, 640, 3]\n" in text
    assert "Outputs: 2\n" in text
    assert "INDETERMINATE_UNSUPPORTED (exit 4)" in text
    assert "Frigate Structural Contract\n" in text


def test_model_metadata_does_not_duplicate_probe_model_path() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(0, [1, 255, 80, 80], "NCHW")],
        model_path="probe-only-internal-path.rknn",
    )
    text = render_human(_compatible_report(
        model_path="/authoritative/cli.rknn",
        configured_model_path="models/from-config.rknn",
        model_metadata=model,
    ))
    metadata_block = text.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract\n", 1)[0]
    assert "probe-only-internal-path.rknn" not in metadata_block
    assert "probe-only-internal-path.rknn" not in text
    assert "  inspected path: /authoritative/cli.rknn\n" in text
    assert "Inputs: 1\n" in metadata_block
    assert "Outputs: 1\n" in metadata_block


def test_structural_findings_render_explicit_pass_fail_and_unknown_comparisons() -> None:
    findings = (
        _finding("input.count", Layer.FRIGATE_STRUCTURAL, Status.PASS, "Exactly one input", expected=1, actual=1),
        _finding("input.height", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "Height differs", expected=640, actual=320),
        _finding(
            "input.dtype", Layer.FRIGATE_STRUCTURAL, Status.UNKNOWN,
            "Input dtype conversion semantics are not mechanically proven",
            expected="FLOAT32", actual="INT8",
        ),
    )
    text = render_human(_compatible_report(findings=findings))
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert (
        "  Input count\n"
        "    Frigate expects\n"
        "      count: 1\n"
        "    RKNN model reports\n"
        "      count: 1\n"
        "    Result\n"
        "      PASS\n"
        "      Exactly one input\n"
    ) in structural
    assert "  Input height\n    Frigate expects\n      height: 640\n" in structural
    assert "    RKNN model reports\n      height: 320\n    Result\n      FAIL\n      Height differs\n" in structural
    assert "  Input dtype\n" in structural
    assert "      dtype: FLOAT32\n" in structural
    assert "      dtype: INT8\n" in structural
    assert "      UNKNOWN\n      Input dtype conversion semantics are not mechanically proven" in structural


def test_structural_input_dimensions_layout_dtype_and_quantization_use_finding_values() -> None:
    values = (
        ("input.rank", 4, 4, Status.PASS),
        ("input.batch", 1, 1, Status.PASS),
        ("input.height", 640, 320, Status.FAIL),
        ("input.width", 640, 640, Status.PASS),
        ("input.channels", 3, 3, Status.PASS),
        ("input.layout", "NHWC", "NCHW", Status.FAIL),
        ("input.dtype", "FLOAT32", "INT8", Status.UNKNOWN),
        ("input.quantization", None, "AFFINE", Status.UNKNOWN),
    )
    findings = tuple(
        _finding(check_id, Layer.FRIGATE_STRUCTURAL, status, f"reason for {check_id}", expected=expected, actual=actual)
        for check_id, expected, actual, status in values
    )
    text = render_human(_compatible_report(findings=findings))
    for check_id, expected, actual, status in values:
        label = check_id.split(".", 1)[1]
        expected_text = "unavailable" if expected is None else str(expected)
        assert f"      {label}: {expected_text}\n" in text
        assert f"      {label}: {actual}\n" in text
        assert f"      {status.value}\n      reason for {check_id}\n" in text


def test_output_index_count_and_two_output_unknown_are_authoritative() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(0, [1, 255, 80, 80], "NCHW"), _tensor(1, [1, 255, 40, 40], "NCHW")],
    )
    findings = (
        _finding(
            "output.index_range", Layer.FRIGATE_STRUCTURAL, Status.FAIL,
            "Multipart output indices must fit the handler maps",
            handler="yolo-generic:multipart", expected="indices 0..2", actual=[0, 1, 2, 3],
        ),
        _finding(
            "output.scale_completeness", Layer.FRIGATE_STRUCTURAL, Status.UNKNOWN,
            "Two outputs are executable but third-scale completeness is unproven",
            handler="yolo-generic:multipart", expected=3, actual=2,
        ),
    )
    text = render_human(_compatible_report(findings=findings, model_metadata=model))
    assert "      supported indices: indices 0..2\n" in text
    assert "      output indices: [0, 1, 2, 3]\n" in text
    assert "      output count: 2\n" in text
    assert "      FAIL\n      Multipart output indices must fit the handler maps\n" in text
    assert "      required output count: 3\n" in text
    assert "      reported output count: 2\n" in text
    assert "      UNKNOWN\n      Two outputs are executable but third-scale completeness is unproven\n" in text


def test_output_rank_layout_and_reshape_include_only_authoritative_values_and_context() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(7, [1, 2, 3], "NCHW")],
    )
    findings = (
        _finding("output.7.rank", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "rank reason", expected=4, actual=3),
        _finding("output.7.layout", Layer.FRIGATE_STRUCTURAL, Status.PASS, "layout reason", expected="NCHW", actual="NCHW"),
        _finding("output.7.reshape", Layer.FRIGATE_STRUCTURAL, Status.PASS, "authoritative reason", expected=10, actual=10),
    )
    text = render_human(_compatible_report(findings=findings, model_metadata=model))
    assert "  Output 7 rank\n" in text
    assert "      rank: 4\n" in text
    assert "      rank: 3\n" in text
    assert "  Output 7 layout\n" in text
    assert "      layout: NCHW\n" in text
    reshape = text.split("  Output 7 reshape\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert "      required elements: 10\n" in reshape
    assert "      shape: [1, 2, 3]\n" in reshape
    assert "      layout: NCHW\n" in reshape
    assert "      actual elements: 10\n" in reshape
    assert "      PASS\n      authoritative reason" in reshape
    assert "required elements: 6" not in reshape
    assert "actual elements: 6" not in reshape


def test_generic_structural_fallback_is_deterministic_and_future_safe() -> None:
    finding = _finding(
        "future.contract.check", Layer.FRIGATE_STRUCTURAL, Status.UNKNOWN, "future reason",
        handler="future:branch",
        expected={"z": (2, 1), "a": True},
        actual={"items": [None, "x"], "enabled": False},
    )
    report = _compatible_report(findings=(finding,))
    first = render_human(report)
    second = render_human(report)
    assert first == second
    assert "  future.contract.check\n" in first
    assert '      value: {"a": true, "z": (2, 1)}\n' in first
    assert '      value: {"enabled": false, "items": [unavailable, "x"]}\n' in first
    assert "      UNKNOWN\n      future reason\n" in first
    assert "evaluated contract: future:branch\n" in first


def test_not_evaluated_and_operation_failure_do_not_fabricate_comparisons() -> None:
    findings = (
        _finding("input.rank", Layer.FRIGATE_STRUCTURAL, Status.NOT_EVALUATED, "not run", expected=4, actual=None),
        _finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "unsupported handler: yolox"),
    )
    text = render_human(_compatible_report(
        overall=Overall.INDETERMINATE_UNSUPPORTED,
        findings=findings,
        handler="yolox",
        model_metadata=None,
    ))
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert "input.rank" not in structural
    assert "Input rank" not in structural
    assert "FAIL operation.failure: unsupported handler: yolox" in structural
    assert "Frigate expects" not in structural
    assert "RKNN model reports" not in structural


def test_semantic_findings_stay_separate_and_metadata_section_is_unchanged() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(0, [1, 255, 80, 80], "NCHW")],
    )
    findings = (
        _finding("input.count", Layer.FRIGATE_STRUCTURAL, Status.PASS, "count reason", expected=1, actual=1),
        _finding(
            "semantic.color",
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            "RGB/BGR semantic channel meaning is not exposed by structural metadata",
        ),
    )
    text = render_human(_compatible_report(findings=findings, model_metadata=model))
    metadata = text.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract", 1)[0]
    assert metadata == (
        "Inputs: 1\nInput 0\n  shape: [1, 640, 640, 3]\n  layout: NHWC\n"
        "  dtype: INT8\n  quantization: AFFINE\nOutputs: 1\nOutput 0\n"
        "  shape: [1, 255, 80, 80]\n  layout: NCHW\n  dtype: INT8\n  quantization: AFFINE"
    )
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert (
        "  Color channel meaning\n"
        "    Result\n"
        "      UNKNOWN\n"
        "      RGB/BGR semantic channel meaning is not exposed by structural metadata"
    ) in semantic
    assert "Frigate expects" not in semantic
    assert "RKNN model reports" not in semantic
    assert "UNKNOWN semantic.color:" not in semantic
    overall = text.split("\nOverall\n", 1)[1]
    assert "STRUCTURALLY_COMPATIBLE (exit 0)" in overall
    assert "does not prove semantic compatibility" in overall
    assert "successful inference" in overall


@pytest.mark.parametrize(
    ("check_id", "label"),
    [
        ("input.count", "Input count"),
        ("input.rank", "Input rank"),
        ("input.batch", "Input batch"),
        ("input.height", "Input height"),
        ("input.width", "Input width"),
        ("input.channels", "Input channels"),
        ("input.layout", "Input layout"),
        ("input.dtype", "Input dtype"),
        ("input.quantization", "Input quantization"),
        ("output.index_range", "Output index range"),
        ("output.scale_completeness", "Output scale completeness"),
    ],
)
def test_known_structural_check_ids_have_stable_human_labels(check_id: str, label: str) -> None:
    assert _STRUCTURAL_LABELS[check_id] == label
    for status in (Status.PASS, Status.FAIL, Status.UNKNOWN):
        finding = _finding(
            check_id,
            Layer.FRIGATE_STRUCTURAL,
            status,
            f"misleading summary for {status.value}: this is Output 99 reshape",
            expected=1,
            actual=2,
        )
        assert _structural_title(finding) == label
        text = render_human(_compatible_report(findings=(finding,)))
        assert f"  {label}\n" in text
        assert f"      {status.value}\n" in text
        assert f"      misleading summary for {status.value}: this is Output 99 reshape\n" in text


def test_structural_label_map_covers_every_non_indexed_checker_id() -> None:
    assert set(_STRUCTURAL_LABELS) == {
        "input.count",
        "input.rank",
        "input.batch",
        "input.height",
        "input.width",
        "input.channels",
        "input.layout",
        "input.dtype",
        "input.quantization",
        "output.index_range",
        "output.scale_completeness",
    }


@pytest.mark.parametrize(
    ("check_id", "label"),
    [
        ("output.0.rank", "Output 0 rank"),
        ("output.0.layout", "Output 0 layout"),
        ("output.0.reshape", "Output 0 reshape"),
        ("output.8.rank", "Output 8 rank"),
        ("output.8.layout", "Output 8 layout"),
        ("output.8.reshape", "Output 8 reshape"),
    ],
)
def test_indexed_output_labels_use_finding_id_index_not_list_position(check_id: str, label: str) -> None:
    finding = _finding(check_id, Layer.FRIGATE_STRUCTURAL, Status.FAIL, "indexed reason", expected=1, actual=0)
    assert _structural_title(finding) == label
    text = render_human(_compatible_report(findings=(finding,)))
    assert f"  {label}\n" in text


def test_indexed_labels_are_independent_of_metadata_and_status() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(0, [1, 1, 1, 1], "NCHW"), _tensor(1, [1, 1, 1, 1], "NCHW")],
    )
    finding = _finding(
        "output.8.reshape",
        Layer.FRIGATE_STRUCTURAL,
        Status.UNKNOWN,
        "Output 0 reshape according to summary",
        expected=9,
        actual=8,
    )
    assert _structural_title(finding) == "Output 8 reshape"
    text = render_human(_compatible_report(findings=(finding,), model_metadata=model))
    assert "  Output 8 reshape\n" in text
    assert "  Output 0 reshape\n" not in text
    assert "      UNKNOWN\n      Output 0 reshape according to summary\n" in text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "unavailable"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (42, "42"),
        (-7, "-7"),
        (1.5, "1.5"),
        (-0.25, "-0.25"),
        ("indices 0..2", "indices 0..2"),
        ("", ""),
        ([3, 1, 2], "[3, 1, 2]"),
        ((3, 1), "(3, 1)"),
        ((1,), "(1,)"),
        ([], "[]"),
        ((), "()"),
        ({}, "{}"),
        ({"z": 1, "a": 2}, '{"a": 2, "z": 1}'),
        ({"b": False, "a": None}, '{"a": unavailable, "b": false}'),
        ({"items": [True, "x"], "n": 1}, '{"items": [true, "x"], "n": 1}'),
        ({"nested": {"b": 2, "a": 1}}, '{"nested": {"a": 1, "b": 2}}'),
    ],
)
def test_stable_value_formatting_is_deterministic(value: object, expected: str) -> None:
    assert _stable_value(value) == expected
    assert _stable_value(value) == _stable_value(value)


def test_sequence_formatting_preserves_authoritative_order() -> None:
    assert _stable_value([9, 0, 8, 1]) == "[9, 0, 8, 1]"
    assert _stable_value((9, 0, 8, 1)) == "(9, 0, 8, 1)"


def test_mapping_formatting_sorts_keys_only() -> None:
    left = {"m": 1, "a": [2, 1], "z": {"y": 1, "x": 0}}
    right = {"z": {"x": 0, "y": 1}, "a": [2, 1], "m": 1}
    assert _stable_value(left) == _stable_value(right)
    assert _stable_value(left) == '{"a": [2, 1], "m": 1, "z": {"x": 0, "y": 1}}'


def test_unknown_finding_fallback_preserves_id_values_status_and_summary() -> None:
    finding = _finding(
        "future.contract.rule",
        Layer.FRIGATE_STRUCTURAL,
        Status.FAIL,
        "verbatim future reason: do not rewrite",
        expected=None,
        actual={"z": True, "a": [False, None]},
    )
    report = _compatible_report(
        overall=Overall.STRUCTURALLY_INCOMPATIBLE,
        findings=(finding,),
    )
    before = (report.overall, finding.check_id, finding.expected, finding.actual, finding.status, finding.summary)
    text = render_human(report)
    after = (report.overall, finding.check_id, finding.expected, finding.actual, finding.status, finding.summary)
    assert before == after
    assert "  future.contract.rule\n" in text
    assert "      value: unavailable\n" in text
    assert '      value: {"a": [false, unavailable], "z": true}\n' in text
    assert "      FAIL\n      verbatim future reason: do not rewrite\n" in text
    assert "Input count" not in text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic", 1)[0]


def test_generic_fallback_value_labels_remain_boring() -> None:
    finding = _finding(
        "brand.new.check",
        Layer.FRIGATE_STRUCTURAL,
        Status.PASS,
        "ok",
        expected="same",
        actual="same",
    )
    text = render_human(_compatible_report(findings=(finding,)))
    block = text.split("  brand.new.check\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert "    Frigate expects\n      value: same\n" in block
    assert "    RKNN model reports\n      value: same\n" in block
    assert "      PASS\n      ok" in block


def test_status_and_summary_are_preserved_verbatim_for_known_labels() -> None:
    summary = "Height differs [do not rewrite] expected!=actual"
    finding = _finding(
        "input.height",
        Layer.FRIGATE_STRUCTURAL,
        Status.FAIL,
        summary,
        expected=640,
        actual=320,
    )
    text = render_human(_compatible_report(findings=(finding,)))
    assert "  Input height\n" in text
    assert "      FAIL\n" in text
    assert f"      {summary}\n" in text
    assert finding.summary == summary
    assert finding.status is Status.FAIL


def test_not_evaluated_is_omitted_from_completed_comparisons() -> None:
    findings = (
        _finding("input.count", Layer.FRIGATE_STRUCTURAL, Status.NOT_EVALUATED, "skipped", expected=1, actual=None),
        _finding("input.rank", Layer.FRIGATE_STRUCTURAL, Status.PASS, "rank ok", expected=4, actual=4),
    )
    text = render_human(_compatible_report(findings=findings))
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert "Input count" not in structural
    assert "NOT_EVALUATED" not in structural
    assert "  Input rank\n" in structural
    assert "      PASS\n      rank ok" in structural


def test_renderer_not_checker_reshape_regression_remains_authoritative() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(7, [1, 2, 3], "NCHW")],
    )
    findings = (
        _finding(
            "output.7.reshape",
            Layer.FRIGATE_STRUCTURAL,
            Status.PASS,
            "authoritative reason",
            expected=10,
            actual=10,
        ),
    )
    text = render_human(_compatible_report(findings=findings, model_metadata=model))
    reshape = text.split("  Output 7 reshape\n", 1)[1].split("\n\nSemantic / Unproven Properties", 1)[0]
    assert "      required elements: 10\n" in reshape
    assert "      actual elements: 10\n" in reshape
    assert "      shape: [1, 2, 3]\n" in reshape
    assert "      PASS\n      authoritative reason" in reshape
    assert "required elements: 6" not in reshape
    assert "actual elements: 6" not in reshape
    assert "6" not in reshape.split("shape: [1, 2, 3]\n", 1)[1]


@pytest.mark.parametrize(
    ("check_id", "label"),
    list(_SEMANTIC_LABELS.items()),
)
def test_known_semantic_labels_are_stable_and_not_parsed_from_summary(check_id: str, label: str) -> None:
    finding = _finding(
        check_id,
        Layer.SEMANTIC_UNCERTAINTY,
        Status.UNKNOWN,
        "misleading summary claiming PASS and YOLOv8 compatibility",
    )
    assert _semantic_title(finding) == label
    text = render_human(_compatible_report(findings=(finding,)))
    assert f"  {label}\n" in text
    assert "      UNKNOWN\n" in text
    assert "      misleading summary claiming PASS and YOLOv8 compatibility\n" in text
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert "Result\n      PASS" not in semantic
    assert "Result\n      FAIL" not in semantic


def test_semantic_label_map_covers_every_current_checker_subject() -> None:
    assert set(_SEMANTIC_LABELS) == {
        "semantic.color",
        "semantic.preprocessing",
        "semantic.dfl",
        "semantic.objectness",
        "semantic.classes",
        "semantic.boxes",
        "semantic.anchors",
        "semantic.labels",
        "semantic.dequantization",
    }


def test_multiple_semantic_unknowns_use_authoritative_reasons() -> None:
    findings = tuple(
        _finding(
            check_id,
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            f"{label} is not exposed by structural metadata",
        )
        for check_id, label in _SEMANTIC_LABELS.items()
    )
    text = render_human(_compatible_report(findings=findings))
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert "  status: UNKNOWN\n" in semantic
    for check_id, label in _SEMANTIC_LABELS.items():
        assert f"  {label}\n" in semantic
        assert f"      {label} is not exposed by structural metadata" in semantic
        assert check_id not in semantic
    assert "Frigate expects" not in semantic
    assert "expected:" not in semantic
    assert "actual:" not in semantic


def test_generic_semantic_fallback_preserves_id_status_and_summary() -> None:
    finding = _finding(
        "semantic.future.property",
        Layer.SEMANTIC_UNCERTAINTY,
        Status.UNKNOWN,
        "future semantic reason verbatim",
        expected="hint",
        actual="observed",
    )
    before = (finding.check_id, finding.status, finding.summary, finding.expected, finding.actual)
    text = render_human(_compatible_report(findings=(finding,)))
    assert before == (finding.check_id, finding.status, finding.summary, finding.expected, finding.actual)
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert "  semantic.future.property\n" in semantic
    assert "    expected: hint\n" in semantic
    assert "    actual: observed\n" in semantic
    assert "      UNKNOWN\n      future semantic reason verbatim" in semantic


def test_unknown_is_distinct_from_not_evaluated_for_semantics() -> None:
    unknown = render_human(_compatible_report(findings=(
        _finding(
            "semantic.color",
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            "RGB/BGR semantic channel meaning is not exposed by structural metadata",
        ),
    )))
    unknown_semantic = unknown.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert "  status: UNKNOWN\n" in unknown_semantic
    assert "  Color channel meaning\n" in unknown_semantic
    assert "      UNKNOWN\n" in unknown_semantic

    not_run = render_human(Report(
        overall=Overall.RUNTIME_MODEL_ERROR,
        findings=(
            _finding("operation.failure", Layer.RUNTIME_MODEL, Status.FAIL, "probe failed"),
        ),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/model.rknn",
        model_metadata=None,
    ))
    not_run_semantic = not_run.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert not_run_semantic == "  status: NOT_EVALUATED"
    assert "Color channel meaning" not in not_run_semantic
    assert "UNKNOWN" not in not_run_semantic
    assert "Overall\n  RUNTIME_MODEL_ERROR (exit 2)\n" in not_run
    assert "Structural and semantic compatibility were not proven because Runtime or model inspection failed." in not_run


def test_unsupported_path_does_not_fabricate_semantic_unknowns() -> None:
    text = render_human(Report(
        overall=Overall.INDETERMINATE_UNSUPPORTED,
        findings=(
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "verified"),
            _finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "unsupported handler: yolox"),
        ),
        selected_detector="rknn",
        handler="yolox",
        model_path="/model.rknn",
        configured_model_path="models/yolox.rknn",
        runtime={"selection_mode": "default"},
        model_metadata=_model([_tensor(0, [1, 640, 640, 3], "NHWC")], [_tensor(0, [1, 1, 1, 1], "NCHW")]),
    ))
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert semantic == "  status: NOT_EVALUATED"
    assert "Color channel meaning" not in text.split("Semantic / Unproven Properties\n", 1)[1]
    assert "Inputs: 1\n" in text
    assert "INDETERMINATE_UNSUPPORTED (exit 4)" in text
    assert "outside the supported mechanical scope" in text
    assert "no compatibility or incompatibility is concluded" in text


@pytest.mark.parametrize(
    ("overall", "exit_code", "needle"),
    [
        (Overall.STRUCTURALLY_COMPATIBLE, 0, "Compatible within the mechanically checked structural scope"),
        (Overall.STRUCTURALLY_INCOMPATIBLE, 1, "mechanically proven structural mismatch was found against the selected supported contract"),
        (Overall.RUNTIME_MODEL_ERROR, 2, "Runtime or model inspection failed"),
        (Overall.INVALID_INPUT, 3, "Input or configuration was invalid"),
        (Overall.INDETERMINATE_UNSUPPORTED, 4, "outside the supported mechanical scope"),
        (Overall.INTERNAL_ERROR, 5, "internal failure prevented a complete evaluation"),
    ],
)
def test_overall_outcome_wording_preserves_enum_and_exit(overall: Overall, exit_code: int, needle: str) -> None:
    report = Report(overall=overall, findings=())
    text = render_human(report)
    assert f"Overall\n  {overall.value} (exit {exit_code})\n" in text
    assert needle in text
    assert report.overall is overall
    assert report.exit_code == exit_code


def test_structurally_compatible_disclaimer_excludes_semantic_and_inference_claims() -> None:
    text = render_human(_compatible_report())
    overall = text.split("\nOverall\n", 1)[1]
    assert "STRUCTURALLY_COMPATIBLE (exit 0)" in overall
    assert "mechanically checked structural scope" in overall
    assert "does not prove semantic compatibility" in overall
    assert "preprocessing correctness" in overall
    assert "successful inference" in overall
    assert "production Frigate integration" in overall


def test_structurally_incompatible_wording_stays_contract_scoped() -> None:
    text = render_human(_compatible_report(
        overall=Overall.STRUCTURALLY_INCOMPATIBLE,
        findings=(
            _finding("input.height", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "Height differs", expected=640, actual=320),
        ),
    ))
    overall = text.split("\nOverall\n", 1)[1]
    assert "STRUCTURALLY_INCOMPATIBLE (exit 1)" in overall
    assert "selected supported contract" in overall
    lowered = overall.lower()
    assert "model itself is invalid" not in lowered
    assert "model is broken" not in lowered
    assert "all frigate handlers" not in lowered


def test_metadata_observations_do_not_become_semantic_claims() -> None:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC", name="yolov8_input")],
        [_tensor(0, [1, 64, 80, 80], "NCHW", name="dfl_objectness")],
    )
    findings = (
        _finding(
            "semantic.dfl",
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            "DFL representation is not exposed by structural metadata",
        ),
        _finding(
            "semantic.dequantization",
            Layer.SEMANTIC_UNCERTAINTY,
            Status.UNKNOWN,
            "RKNN Lite numerical/dequantization behavior is not exposed by structural metadata",
        ),
    )
    text = render_human(_compatible_report(
        findings=findings,
        model_metadata=model,
        model_path="/models/yolov8m_int8.rknn",
    ))
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall", 1)[0]
    assert "DFL representation" in semantic
    assert "RKNN Lite dequantization behavior" in semantic
    assert "      UNKNOWN\n" in semantic
    assert "PASS" not in semantic
    assert "FAIL" not in semantic
    assert "yolov8" not in semantic.lower()
    assert "INT8" not in semantic
    assert "AFFINE" not in semantic
    assert "compatible" not in semantic.lower()
