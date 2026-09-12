from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import tensor
from rknn_frigate_compat.checks import common_input_checks, multipart_checks, semantic_findings
from rknn_frigate_compat.config import ConfigError, classify_model_path, load_config
from rknn_frigate_compat.models import (
    ErrorSubtype,
    Finding,
    FrigateConfig,
    Layer,
    ModelMetadata,
    Overall,
    Report,
    Severity,
    Status,
    TensorMetadata,
    aggregate,
)


FIXTURES = Path(__file__).parent / "fixtures" / "frigate"


def metadata(inputs, outputs) -> ModelMetadata:
    def convert(value) -> TensorMetadata:
        return TensorMetadata(
            index=value["index"], name=value["name"], n_dims=value["n_dims"], dims=tuple(value["dims"]),
            format=value["format"], type=value["type"], quantization_type=value["quantization_type"],
            zero_point=value["zero_point"], scale=value["scale"], element_count=value["element_count"],
            byte_size=value["byte_size"], width_stride=value["width_stride"],
            byte_size_with_stride=value["byte_size_with_stride"],
        )
    return ModelMetadata("synthetic.rknn", len(inputs), len(outputs), tuple(map(convert, inputs)), tuple(map(convert, outputs)))


def config(**changes) -> FrigateConfig:
    value = FrigateConfig("rknn", "models/test.rknn", "custom", "yolo-generic", 640, 640, "nhwc", "rgb", "int", {})
    return replace(value, **changes)


def test_result_enums_serialization_and_precedence() -> None:
    finding = Finding("x", Layer.SEMANTIC_UNCERTAINTY, None, Severity.WARNING, Status.UNKNOWN, "unknown")
    report = Report(Overall.STRUCTURALLY_COMPATIBLE, (finding,), error_subtype=ErrorSubtype.RUNTIME_SELECTION_ERROR)
    assert report.exit_code == 0
    assert report.data()["findings"][0]["status"] == "UNKNOWN"
    assert aggregate([Overall.STRUCTURALLY_COMPATIBLE, Overall.RUNTIME_MODEL_ERROR]) is Overall.RUNTIME_MODEL_ERROR
    assert aggregate([Overall.RUNTIME_MODEL_ERROR, Overall.INTERNAL_ERROR]) is Overall.INTERNAL_ERROR
    assert aggregate([Overall.INTERNAL_ERROR, Overall.INVALID_INPUT]) is Overall.INVALID_INPUT


def test_report_model_metadata_is_additive_focused_and_deterministic() -> None:
    first = tensor(0, [1, 255, 80, 80], "NCHW")
    second = tensor(1, [1, 255, 40, 40], "NCHW")
    model = metadata([tensor(0, [1, 640, 640, 3], "NHWC")], [second, first])
    finding = Finding(
        "x", Layer.SEMANTIC_UNCERTAINTY, None, Severity.WARNING,
        Status.UNKNOWN, "unknown", None, "observed", ("source",),
    )
    report = Report(
        Overall.STRUCTURALLY_COMPATIBLE,
        (finding,),
        selected_detector="rknn",
        handler="yolo-generic",
        model_path="/authoritative/cli-model.rknn",
        configured_model_path="models/configured.rknn",
        runtime={"selection_mode": "default"},
        model_metadata=model,
    )

    assert report.model_metadata is model
    data = report.data()
    legacy = {key: value for key, value in data.items() if key != "model_metadata"}
    assert legacy == {
        "schema_version": 1,
        "overall": "STRUCTURALLY_COMPATIBLE",
        "exit_code": 0,
        "selected_detector": "rknn",
        "handler": "yolo-generic",
        "model_path": "/authoritative/cli-model.rknn",
        "configured_model_path": "models/configured.rknn",
        "runtime": {"selection_mode": "default"},
        "error_subtype": None,
        "layers": {
            "runtime_model": "NOT_EVALUATED",
            "frigate_structural": "NOT_EVALUATED",
            "semantic_uncertainty": "UNKNOWN",
        },
        "findings": [{
            "check_id": "x",
            "layer": "semantic_uncertainty",
            "handler": None,
            "severity": "WARNING",
            "status": "UNKNOWN",
            "summary": "unknown",
            "expected": None,
            "actual": "observed",
            "evidence": ["source"],
        }],
    }
    observed = data["model_metadata"]
    assert observed is not None
    assert set(observed) == {"input_count", "inputs", "output_count", "outputs"}
    assert "model_path" not in observed
    assert [item["index"] for item in observed["inputs"]] == [0]
    assert [item["index"] for item in observed["outputs"]] == [0, 1]
    assert observed["outputs"][0] == {
        "index": 0,
        "name": "tensor0",
        "n_dims": 4,
        "dims": (1, 255, 80, 80),
        "format": "NCHW",
        "type": "INT8",
        "quantization_type": "AFFINE",
        "zero_point": -1,
        "scale": 0.1,
        "element_count": 1632000,
        "byte_size": 1632000,
        "width_stride": 80,
        "byte_size_with_stride": 1632000,
        "fractional_length": None,
    }


def test_report_without_inspection_serializes_null_model_metadata() -> None:
    assert Report(Overall.INTERNAL_ERROR).data()["model_metadata"] is None


def test_config_unique_defaults_and_unrelated_sections() -> None:
    selected = load_config(FIXTURES / "defaults.yaml")
    assert selected.detector == "rknn"
    assert selected.model_type == "ssd"
    assert selected.width == selected.height == 320
    assert set(selected.provenance.values()) == {"default"}
    explicit = load_config(FIXTURES / "supported.yaml")
    assert explicit.model_path_kind == "custom"
    assert explicit.width == 640
    assert set(explicit.provenance.values()) == {"explicit"}


def test_detector_disambiguation_and_failures() -> None:
    with pytest.raises(ConfigError, match="multiple RKNN"):
        load_config(FIXTURES / "ambiguous.yaml")
    assert load_config(FIXTURES / "ambiguous.yaml", "second").detector == "second"
    with pytest.raises(ConfigError, match="not RKNN"):
        load_config(FIXTURES / "ambiguous.yaml", "cpu")
    with pytest.raises(ConfigError, match="unknown detector"):
        load_config(FIXTURES / "ambiguous.yaml", "missing")
    with pytest.raises(ConfigError, match="no RKNN"):
        load_config(FIXTURES / "missing-rknn.yaml")


def test_multiple_detectors_with_one_rknn_selects_it(tmp_path: Path) -> None:
    path = tmp_path / "multiple.yaml"
    path.write_text("detectors: {cpu: {type: cpu}, accelerator: {type: rknn}}\nmodel: {path: models/a.rknn}\n", encoding="utf-8")
    assert load_config(path).detector == "accelerator"


@pytest.mark.parametrize(
    ("value", "kind"),
    [(None, "unsupported_missing"), ("plus://x", "unsupported_plus"), ("yolov8n-320", "unsupported_preset"),
     ("models/a.rknn", "custom"), ("./a.rknn", "custom"), ("/models/a.rknn", "custom")],
)
def test_model_path_classification(value: str | None, kind: str) -> None:
    assert classify_model_path(value) == kind


def test_malformed_relevant_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("detectors: []\nmodel: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
    path.write_text("detectors: {r: {type: rknn}}\nmodel: {path: models/a.rknn, width: true}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("name", ["supported.yaml", "defaults.yaml", "unsupported-handler.yaml", "preset.yaml", "plus.yaml", "input-dimension-mismatch.yaml", "input-layout-mismatch.yaml", "rk3588-yolov8m.yaml"])
def test_each_nonambiguous_yaml_fixture_loads(name: str) -> None:
    selected = load_config(FIXTURES / name)
    assert selected.detector == "rknn"


def test_no_rknn_binary_is_committed_as_fixture() -> None:
    assert not list((Path(__file__).parent / "fixtures").rglob("*.rknn"))


@pytest.mark.parametrize(
    ("dims", "fmt", "field", "expected_status"),
    [
        ([1, 640, 640, 3], "NHWC", "input.height", Status.PASS),
        ([1, 320, 640, 3], "NHWC", "input.height", Status.FAIL),
        ([1, 3, 640, 640], "NCHW", "input.layout", Status.FAIL),
        ([1, 640, 640, 3], "UNKNOWN", "input.height", Status.UNKNOWN),
    ],
)
def test_common_input_boundaries(dims, fmt: str, field: str, expected_status: Status) -> None:
    findings = common_input_checks(config(), metadata([tensor(0, dims, fmt)], [tensor(0, [1, 255, 80, 80], "NCHW"), tensor(1, [1, 255, 40, 40], "NCHW")]))
    assert next(item.status for item in findings if item.check_id == field) is expected_status
    assert not any(item.check_id in {"input.dtype", "input.quantization"} and item.status is Status.FAIL for item in findings)


def test_input_count_and_rank_fail_deterministically() -> None:
    assert common_input_checks(config(), metadata([], []))[0].status is Status.FAIL
    findings = common_input_checks(config(), metadata([tensor(0, [1, 640, 3], "NHWC")], []))
    assert [item.check_id for item in findings] == ["input.count", "input.rank"]


def test_multipart_pass_two_output_unknown_and_failures() -> None:
    good = metadata([tensor(0, [1, 640, 640, 3], "NHWC")], [
        tensor(0, [1, 255, 80, 80], "NCHW"), tensor(1, [1, 255, 40, 40], "NCHW")])
    findings = multipart_checks(good)
    assert not any(item.status is Status.FAIL for item in findings)
    assert next(item for item in findings if item.check_id == "output.scale_completeness").status is Status.UNKNOWN
    nine = [tensor(i, [1, 64, 80, 80], "NCHW") for i in range(9)]
    failures = multipart_checks(metadata([tensor(0, [1, 640, 640, 3], "NHWC")], nine))
    assert next(item for item in failures if item.check_id == "output.index_range").status is Status.FAIL
    reshape = next(item for item in failures if item.check_id == "output.0.reshape")
    assert reshape.expected == 1632000 and reshape.actual == 409600 and reshape.status is Status.FAIL


def test_output_rank_and_layout_ordering() -> None:
    rank = multipart_checks(metadata([], [tensor(0, [1, 255, 80], "NCHW"), tensor(1, [1, 255, 40, 40], "NCHW")]))
    assert not any(item.check_id == "output.0.reshape" for item in rank)
    layout = multipart_checks(metadata([], [tensor(0, [1, 80, 80, 255], "NHWC"), tensor(1, [1, 40, 40, 255], "NHWC")]))
    assert next(item for item in layout if item.check_id == "output.0.layout").status is Status.FAIL


def test_all_semantic_findings_are_non_failing() -> None:
    findings = semantic_findings(config())
    assert len(findings) == 9
    assert {item.status for item in findings} == {Status.UNKNOWN}
