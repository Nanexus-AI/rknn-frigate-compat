"""Deterministic human-report content and section-order snapshot tests."""

from __future__ import annotations

from pathlib import Path

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
from rknn_frigate_compat.report import _SEMANTIC_LABELS, render_human, render_json


GOLDENS = Path(__file__).parent / "fixtures" / "report" / "human"

SECTION_HEADERS = (
    "Selection / Environment",
    "RKNN Model Metadata",
    "Frigate Structural Contract",
    "Semantic / Unproven Properties",
    "Overall",
)

SELECTION_GROUPS = ("Detector", "Handler", "Model", "RKNN Runtime")


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


def _tensor(index: int, dims: list[int], fmt: str) -> TensorMetadata:
    count = 1
    for value in dims:
        count *= value
    return TensorMetadata(
        index=index,
        name=f"tensor{index}",
        n_dims=len(dims),
        dims=tuple(dims),
        format=fmt,
        type="INT8",
        quantization_type="AFFINE",
        zero_point=-1,
        scale=0.1,
        element_count=count,
        byte_size=count,
        width_stride=dims[-2] if len(dims) >= 2 else 0,
        byte_size_with_stride=count,
    )


def _model(inputs: list[TensorMetadata], outputs: list[TensorMetadata]) -> ModelMetadata:
    return ModelMetadata("probe-internal.rknn", len(inputs), len(outputs), tuple(inputs), tuple(outputs))


DEFAULT_RUNTIME = {
    "selection_mode": "default",
    "inherited_loader_environment": False,
    "loaded_library_path": "/lib/librknnrt.so",
    "loaded_library_sha256": "a" * 64,
    "api_version": "2.3.2",
    "driver_version": "0.9.8",
}

EXPLICIT_RUNTIME = {
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
}


def _assert_section_order(text: str) -> None:
    positions = [text.index(f"{header}\n") for header in SECTION_HEADERS]
    assert positions == sorted(positions)
    assert text.startswith("Selection / Environment\n")
    assert text.endswith("\n")
    assert text.count("\nSelection / Environment\n") == 0  # only the leading heading


def _assert_selection_group_order(text: str) -> None:
    block = text.split("Selection / Environment\n", 1)[1].split("\n\nRKNN Model Metadata\n", 1)[0]
    positions = [block.index(f"{group}\n") for group in SELECTION_GROUPS]
    assert positions == sorted(positions)


def structurally_compatible_report() -> Report:
    model = _model(
        [_tensor(0, [1, 640, 640, 3], "NHWC")],
        [_tensor(1, [1, 255, 40, 40], "NCHW"), _tensor(0, [1, 255, 80, 80], "NCHW")],
    )
    findings = (
        _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified"),
        _finding("input.count", Layer.FRIGATE_STRUCTURAL, Status.PASS, "RKNN detection path supplies exactly one input", expected=1, actual=1),
        _finding("input.rank", Layer.FRIGATE_STRUCTURAL, Status.PASS, "Detection input must be rank four", expected=4, actual=4),
        _finding("input.layout", Layer.FRIGATE_STRUCTURAL, Status.PASS, "Runtime layout must match Frigate input_tensor", expected="NHWC", actual="NHWC"),
        _finding(
            "output.index_range", Layer.FRIGATE_STRUCTURAL, Status.PASS,
            "Multipart output indices must fit the handler maps",
            handler="yolo-generic:multipart", expected="indices 0..2", actual=[0, 1],
        ),
        _finding(
            "output.0.reshape", Layer.FRIGATE_STRUCTURAL, Status.PASS,
            "Output must satisfy the handler reshape element equation",
            handler="yolo-generic:multipart", expected=1632000, actual=1632000,
        ),
        _finding(
            "output.1.reshape", Layer.FRIGATE_STRUCTURAL, Status.PASS,
            "Output must satisfy the handler reshape element equation",
            handler="yolo-generic:multipart", expected=408000, actual=408000,
        ),
        _finding(
            "output.scale_completeness", Layer.FRIGATE_STRUCTURAL, Status.UNKNOWN,
            "Two outputs are executable but third-scale completeness is unproven",
            handler="yolo-generic:multipart", expected=3, actual=2,
        ),
    ) + tuple(
        _finding(
            check_id, Layer.SEMANTIC_UNCERTAINTY, Status.UNKNOWN,
            f"{label} is not exposed by structural metadata",
            handler="yolo-generic",
        )
        for check_id, label in _SEMANTIC_LABELS.items()
    )
    return Report(
        Overall.STRUCTURALLY_COMPATIBLE,
        findings,
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/models/inspected.rknn",
        configured_model_path="models/configured.rknn",
        runtime=DEFAULT_RUNTIME,
        model_metadata=model,
    )


def structurally_incompatible_report() -> Report:
    return Report(
        Overall.STRUCTURALLY_INCOMPATIBLE,
        (
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified"),
            _finding(
                "input.height", Layer.FRIGATE_STRUCTURAL, Status.FAIL,
                "Input height must match Frigate configuration", expected=640, actual=320,
            ),
            _finding(
                "semantic.color", Layer.SEMANTIC_UNCERTAINTY, Status.UNKNOWN,
                "RGB/BGR semantic channel meaning is not exposed by structural metadata",
                handler="yolo-generic",
            ),
        ),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/models/inspected.rknn",
        configured_model_path="models/configured.rknn",
        runtime=DEFAULT_RUNTIME,
        model_metadata=_model(
            [_tensor(0, [1, 320, 320, 3], "NHWC")],
            [_tensor(0, [1, 255, 80, 80], "NCHW"), _tensor(1, [1, 255, 40, 40], "NCHW")],
        ),
    )


def runtime_model_error_report() -> Report:
    return Report(
        Overall.RUNTIME_MODEL_ERROR,
        (
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified"),
            _finding("operation.failure", Layer.RUNTIME_MODEL, Status.FAIL, "probe rknn_init: SYNTHETIC: synthetic failure"),
        ),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/models/inspected.rknn",
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


def unsupported_report() -> Report:
    return Report(
        Overall.INDETERMINATE_UNSUPPORTED,
        (
            _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified"),
            _finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "unsupported handler: yolox"),
        ),
        selected_detector="rknn",
        handler="yolox",
        model_path="/models/inspected.rknn",
        configured_model_path="models/yolox.rknn",
        runtime=DEFAULT_RUNTIME,
        model_metadata=_model(
            [_tensor(0, [1, 640, 640, 3], "NHWC")],
            [_tensor(0, [1, 255, 80, 80], "NCHW"), _tensor(1, [1, 255, 40, 40], "NCHW")],
        ),
    )


def nine_output_explicit_runtime_report() -> Report:
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
    findings = (
        _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified"),
        _finding("input.count", Layer.FRIGATE_STRUCTURAL, Status.PASS, "RKNN detection path supplies exactly one input", expected=1, actual=1),
        _finding(
            "output.index_range", Layer.FRIGATE_STRUCTURAL, Status.FAIL,
            "Multipart output indices must fit the handler maps",
            handler="yolo-generic:multipart", expected="indices 0..2", actual=list(range(9)),
        ),
        _finding(
            "output.0.reshape", Layer.FRIGATE_STRUCTURAL, Status.FAIL,
            "Output must satisfy the handler reshape element equation",
            handler="yolo-generic:multipart", expected=1632000, actual=409600,
        ),
        _finding(
            "output.8.reshape", Layer.FRIGATE_STRUCTURAL, Status.FAIL,
            "Output must satisfy the handler reshape element equation",
            handler="yolo-generic:multipart", expected=102000, actual=25600,
        ),
        _finding(
            "semantic.color", Layer.SEMANTIC_UNCERTAINTY, Status.UNKNOWN,
            "RGB/BGR semantic channel meaning is not exposed by structural metadata",
            handler="yolo-generic",
        ),
        _finding(
            "semantic.dfl", Layer.SEMANTIC_UNCERTAINTY, Status.UNKNOWN,
            "DFL representation is not exposed by structural metadata",
            handler="yolo-generic",
        ),
        _finding(
            "semantic.dequantization", Layer.SEMANTIC_UNCERTAINTY, Status.UNKNOWN,
            "RKNN Lite numerical/dequantization behavior is not exposed by structural metadata",
            handler="yolo-generic",
        ),
    )
    return Report(
        Overall.STRUCTURALLY_INCOMPATIBLE,
        findings,
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/models/nine-output.rknn",
        configured_model_path="models/nine-output.rknn",
        runtime=EXPLICIT_RUNTIME,
        model_metadata=_model([_tensor(0, [1, 640, 640, 3], "NHWC")], outputs),
    )


SNAPSHOT_CASES = (
    ("structurally-compatible.txt", structurally_compatible_report),
    ("structurally-incompatible.txt", structurally_incompatible_report),
    ("runtime-model-error.txt", runtime_model_error_report),
    ("unsupported.txt", unsupported_report),
    ("nine-output-explicit-runtime.txt", nine_output_explicit_runtime_report),
)


@pytest.mark.parametrize(("golden_name", "factory"), SNAPSHOT_CASES)
def test_human_report_matches_golden_fixture(golden_name: str, factory) -> None:
    report = factory()
    text = render_human(report)
    expected = (GOLDENS / golden_name).read_text(encoding="utf-8")
    assert text == expected
    _assert_section_order(text)
    _assert_selection_group_order(text)


def test_canonical_section_order_is_locked_across_paths() -> None:
    for _name, factory in SNAPSHOT_CASES:
        _assert_section_order(render_human(factory()))


def test_compatible_snapshot_locks_scope_disclaimer_and_semantic_unknowns() -> None:
    text = render_human(structurally_compatible_report())
    assert "STRUCTURALLY_COMPATIBLE (exit 0)" in text
    assert "Compatible within the mechanically checked structural scope" in text
    assert "does not prove semantic compatibility" in text
    assert "successful inference" in text
    assert "production Frigate integration" in text
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall\n", 1)[0]
    assert "  status: UNKNOWN\n" in semantic
    labels = list(_SEMANTIC_LABELS.values())
    positions = [semantic.index(f"  {label}\n") for label in labels]
    assert positions == sorted(positions)
    assert "NOT_EVALUATED" not in semantic


def test_incompatible_snapshot_is_contract_scoped() -> None:
    text = render_human(structurally_incompatible_report())
    assert "STRUCTURALLY_INCOMPATIBLE (exit 1)" in text
    assert "selected supported contract" in text
    assert "model itself is invalid" not in text.lower()
    assert "model is broken" not in text.lower()
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties\n", 1)[0]
    assert "  Input height\n" in structural
    assert "      height: 640\n" in structural
    assert "      height: 320\n" in structural
    assert "      FAIL\n" in structural


def test_runtime_error_snapshot_keeps_unavailable_and_not_evaluated() -> None:
    text = render_human(runtime_model_error_report())
    assert "RKNN Model Metadata\n  unavailable\n" in text
    assert "Frigate Structural Contract\n  status: NOT_EVALUATED\n" in text
    assert "Semantic / Unproven Properties\n  status: NOT_EVALUATED\n" in text
    assert "Color channel meaning" not in text
    assert "Frigate expects" not in text
    assert "RUNTIME_MODEL_ERROR (exit 2)" in text
    assert "Runtime or model inspection failed" in text


def test_unsupported_snapshot_preserves_metadata_without_semantic_unknowns() -> None:
    text = render_human(unsupported_report())
    assert "Handler\n  selected: yolox\n" in text
    assert "Inputs: 1\n" in text
    assert "Semantic / Unproven Properties\n  status: NOT_EVALUATED\n" in text
    assert "Color channel meaning" not in text.split("Semantic / Unproven Properties\n", 1)[1]
    assert "INDETERMINATE_UNSUPPORTED (exit 4)" in text
    assert "no compatibility or incompatibility is concluded" in text


def test_nine_output_snapshot_orders_tensors_and_indexed_findings() -> None:
    text = render_human(nine_output_explicit_runtime_report())
    metadata = text.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract\n", 1)[0]
    assert metadata.startswith("Inputs: 1\n")
    assert "Outputs: 9\n" in metadata
    assert metadata.index("Inputs: 1\n") < metadata.index("Outputs: 9\n")
    output_positions = [metadata.index(f"Output {index}\n") for index in range(9)]
    assert output_positions == sorted(output_positions)
    assert "  selection mode: explicit\n" in text
    assert "  identity verification: PASS\n" in text
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties\n", 1)[0]
    assert structural.index("  Output index range\n") < structural.index("  Output 0 reshape\n")
    assert structural.index("  Output 0 reshape\n") < structural.index("  Output 8 reshape\n")
    assert "      output count: 9\n" in structural
    assert "      required elements: 1632000\n" in structural
    assert "      actual elements: 409600\n" in structural
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall\n", 1)[0]
    assert semantic.index("  Color channel meaning\n") < semantic.index("  DFL representation\n")
    assert semantic.index("  DFL representation\n") < semantic.index("  RKNN Lite dequantization behavior\n")
    assert "      UNKNOWN\n" in semantic
    assert "STRUCTURALLY_INCOMPATIBLE (exit 1)" in text


def test_structural_and_semantic_finding_order_follows_report_findings() -> None:
    text = render_human(structurally_compatible_report())
    structural = text.split("Frigate Structural Contract\n", 1)[1].split("\n\nSemantic / Unproven Properties\n", 1)[0]
    assert structural.index("  Input count\n") < structural.index("  Input rank\n")
    assert structural.index("  Input rank\n") < structural.index("  Input layout\n")
    assert structural.index("  Input layout\n") < structural.index("  Output index range\n")
    assert structural.index("  Output 0 reshape\n") < structural.index("  Output 1 reshape\n")
    assert structural.index("  Output 1 reshape\n") < structural.index("  Output scale completeness\n")
    semantic = text.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall\n", 1)[0]
    labels = list(_SEMANTIC_LABELS.values())
    positions = [semantic.index(f"  {label}\n") for label in labels]
    assert positions == sorted(positions)


def test_snapshot_human_values_match_report_authority() -> None:
    report = nine_output_explicit_runtime_report()
    text = render_human(report)
    data = report.data()
    assert data["overall"] == "STRUCTURALLY_INCOMPATIBLE"
    assert data["exit_code"] == 1
    assert f"{data['overall']} (exit {data['exit_code']})" in text
    assert data["model_metadata"]["output_count"] == 9
    assert "Outputs: 9\n" in text
    findings = {item["check_id"]: item for item in data["findings"]}
    assert findings["output.index_range"]["status"] == "FAIL"
    assert findings["output.index_range"]["expected"] == "indices 0..2"
    assert findings["output.index_range"]["actual"] == list(range(9))
    assert "      supported indices: indices 0..2\n" in text
    assert "      output indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]\n" in text
    assert findings["semantic.color"]["status"] == "UNKNOWN"
    assert findings["semantic.color"]["summary"] in text
    # JSON remains free of human section prose.
    machine = render_json(report)
    assert "Selection / Environment" not in machine
    assert "Frigate expects" not in machine
    assert "Color channel meaning" not in machine


def test_unknown_versus_not_evaluated_remain_distinct_in_snapshots() -> None:
    unknown = render_human(structurally_compatible_report())
    not_run = render_human(runtime_model_error_report())
    unknown_semantic = unknown.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall\n", 1)[0]
    not_run_semantic = not_run.split("Semantic / Unproven Properties\n", 1)[1].split("\n\nOverall\n", 1)[0]
    assert "  status: UNKNOWN\n" in unknown_semantic
    assert "Color channel meaning" in unknown_semantic
    assert not_run_semantic == "  status: NOT_EVALUATED"
    assert "Color channel meaning" not in not_run_semantic
