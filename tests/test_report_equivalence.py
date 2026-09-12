"""Human/JSON substantive-equivalence tests for report renderers."""

from __future__ import annotations

import json

import pytest

from rknn_frigate_compat.models import Layer, Overall, Report, Status
from rknn_frigate_compat.report import (
    _SEMANTIC_LABELS,
    _stable_value,
    _structural_title,
    render_human,
    render_json,
)
from test_report_snapshots import (
    DEFAULT_RUNTIME,
    EXPLICIT_RUNTIME,
    nine_output_explicit_runtime_report,
    runtime_model_error_report,
    structurally_compatible_report,
    structurally_incompatible_report,
    unsupported_report,
    _finding,
)


HUMAN_PROSE_MARKERS = (
    "Selection / Environment",
    "Frigate expects",
    "RKNN model reports",
    "Compatible within the mechanically checked structural scope",
    "does not prove semantic compatibility",
    "production Frigate integration",
)

LEGACY_JSON_KEYS = {
    "schema_version",
    "overall",
    "exit_code",
    "selected_detector",
    "handler",
    "model_path",
    "configured_model_path",
    "runtime",
    "error_subtype",
    "layers",
    "findings",
}

RUNTIME_HUMAN_LABELS = {
    "selection_mode": "selection mode",
    "requested_directory": "requested directory",
    "expected_loader_target": "expected loader target",
    "expected_loader_sha256": "expected loader sha256",
    "loaded_library_path": "loaded library path",
    "loaded_library_sha256": "loaded library sha256",
    "identity_verification": "identity verification",
    "api_version": "API version",
    "driver_version": "driver version",
    "inherited_loader_environment": "inherited loader environment",
}


def _shape(dims) -> str:
    return "[" + ", ".join(str(item) for item in dims) + "]"


def _assert_no_human_prose_in_json(payload: str, data: dict) -> None:
    for marker in HUMAN_PROSE_MARKERS:
        assert marker not in payload
    assert "model_metadata" in data
    assert set(LEGACY_JSON_KEYS).issubset(data)


def _assert_selection_equivalence(report: Report, human: str, data: dict) -> None:
    assert data["selected_detector"] == report.selected_detector
    assert data["handler"] == report.handler
    assert data["model_path"] == report.model_path
    assert data["configured_model_path"] == report.configured_model_path
    if report.selected_detector is not None:
        assert f"  selected: {report.selected_detector}\n" in human
    if report.handler is not None:
        assert f"Handler\n  selected: {report.handler}\n" in human
    if report.configured_model_path is not None:
        assert f"  configured path: {report.configured_model_path}\n" in human
    if report.model_path is not None:
        assert f"  inspected path: {report.model_path}\n" in human
    configured = report.handler
    seen: set[str] = set()
    for finding in report.findings:
        handler = finding.handler
        if handler is None or handler == configured or handler in seen:
            continue
        seen.add(handler)
        assert f"  evaluated contract: {handler}\n" in human


def _assert_runtime_equivalence(report: Report, human: str, data: dict) -> None:
    assert data["runtime"] == report.runtime
    runtime = report.runtime
    if not runtime:
        assert "RKNN Runtime\n  unavailable\n" in human
        return
    block = human.split("RKNN Runtime\n", 1)[1].split("\n\n", 1)[0]
    for key, label in RUNTIME_HUMAN_LABELS.items():
        if key not in runtime:
            assert f"  {label}:" not in block
            continue
        value = runtime[key]
        if key == "identity_verification" and value is None:
            assert "identity verification" not in block
            continue
        rendered = "unavailable" if value is None else str(value)
        assert f"  {label}: {rendered}\n" in human
    if runtime.get("selection_mode") == "default":
        assert "requested directory" not in block
        assert "expected loader target" not in block
        assert "identity verification" not in block


def _assert_metadata_equivalence(report: Report, human: str, data: dict) -> None:
    observed = data["model_metadata"]
    if report.model_metadata is None:
        assert observed is None
        assert "RKNN Model Metadata\n  unavailable\n" in human
        return
    assert observed is not None
    assert set(observed) == {"input_count", "inputs", "output_count", "outputs"}
    assert "model_path" not in observed
    assert observed["input_count"] == report.model_metadata.input_count
    assert observed["output_count"] == report.model_metadata.output_count
    block = human.split("RKNN Model Metadata\n", 1)[1].split("\n\nFrigate Structural Contract\n", 1)[0]
    assert f"Inputs: {observed['input_count']}\n" in block
    assert f"Outputs: {observed['output_count']}\n" in block
    assert [item["index"] for item in observed["inputs"]] == sorted(item["index"] for item in observed["inputs"])
    assert [item["index"] for item in observed["outputs"]] == sorted(item["index"] for item in observed["outputs"])
    for kind, tensors in (("Input", observed["inputs"]), ("Output", observed["outputs"])):
        for tensor in tensors:
            snippet = (
                f"{kind} {tensor['index']}\n"
                f"  shape: {_shape(tensor['dims'])}\n"
                f"  layout: {tensor['format']}\n"
                f"  dtype: {tensor['type']}\n"
                f"  quantization: {tensor['quantization_type']}"
            )
            assert snippet in block
    assert report.model_metadata.model_path not in human
    assert report.model_metadata.model_path not in render_json(report)


def _assert_finding_equivalence(report: Report, human: str, data: dict) -> None:
    by_id = {item["check_id"]: item for item in data["findings"]}
    assert len(by_id) == len(data["findings"])
    for finding in report.findings:
        payload = by_id[finding.check_id]
        assert payload["status"] == finding.status.value
        assert payload["summary"] == finding.summary
        assert payload["expected"] == finding.expected
        assert payload["actual"] == finding.actual
        assert payload["handler"] == finding.handler
        assert payload["layer"] == finding.layer.value
        if finding.status is Status.NOT_EVALUATED:
            continue
        if finding.layer is Layer.FRIGATE_STRUCTURAL:
            if finding.check_id == "operation.failure":
                assert f"FAIL operation.failure: {finding.summary}" in human
                continue
            title = _structural_title(finding)
            assert f"  {title}\n" in human
            assert f"      {finding.status.value}\n" in human
            assert finding.summary in human
            assert f"{_stable_value(finding.expected)}" in human
            if finding.actual is not None:
                assert _stable_value(finding.actual) in human
        elif finding.layer is Layer.SEMANTIC_UNCERTAINTY:
            title = _SEMANTIC_LABELS.get(finding.check_id, finding.check_id)
            assert f"  {title}\n" in human
            assert f"      {finding.status.value}\n" in human
            assert finding.summary in human


def _assert_layer_status_equivalence(report: Report, human: str, data: dict) -> None:
    layers = data["layers"]
    assert f"Frigate Structural Contract\n  status: {layers['frigate_structural']}\n" in human
    assert f"Semantic / Unproven Properties\n  status: {layers['semantic_uncertainty']}\n" in human
    runtime_block = human.split("RKNN Runtime\n", 1)[1].split("\n\n", 1)[0]
    assert f"  status: {layers['runtime_model']}" in runtime_block


def _assert_overall_equivalence(report: Report, human: str, data: dict) -> None:
    assert data["schema_version"] == 1
    assert data["overall"] == report.overall.value
    assert data["exit_code"] == report.exit_code
    assert f"Overall\n  {report.overall.value} (exit {report.exit_code})\n" in human


def _assert_substantive_equivalence(report: Report) -> dict:
    human = render_human(report)
    payload = render_json(report)
    data = json.loads(payload)
    assert human == render_human(report)
    assert payload == render_json(report)
    _assert_no_human_prose_in_json(payload, data)
    _assert_selection_equivalence(report, human, data)
    _assert_runtime_equivalence(report, human, data)
    _assert_metadata_equivalence(report, human, data)
    _assert_finding_equivalence(report, human, data)
    _assert_layer_status_equivalence(report, human, data)
    _assert_overall_equivalence(report, human, data)
    assert data["model_path"] == report.model_path
    return data


@pytest.mark.parametrize(
    "factory",
    [
        structurally_compatible_report,
        structurally_incompatible_report,
        runtime_model_error_report,
        unsupported_report,
        nine_output_explicit_runtime_report,
    ],
)
def test_human_json_substantive_equivalence_for_representative_paths(factory) -> None:
    _assert_substantive_equivalence(factory())


def test_compatible_pass_fail_unknown_statuses_agree() -> None:
    report = structurally_compatible_report()
    data = _assert_substantive_equivalence(report)
    findings = {item["check_id"]: item for item in data["findings"]}
    human = render_human(report)
    assert findings["input.count"]["status"] == "PASS"
    assert "  Input count\n" in human
    assert findings["output.scale_completeness"]["status"] == "UNKNOWN"
    assert "  Output scale completeness\n" in human
    assert findings["semantic.color"]["status"] == "UNKNOWN"
    assert "  Color channel meaning\n" in human
    assert data["overall"] == "STRUCTURALLY_COMPATIBLE"
    assert data["exit_code"] == 0


def test_incompatible_fail_and_unknown_agree() -> None:
    report = structurally_incompatible_report()
    data = _assert_substantive_equivalence(report)
    findings = {item["check_id"]: item for item in data["findings"]}
    human = render_human(report)
    assert findings["input.height"]["status"] == "FAIL"
    assert findings["input.height"]["expected"] == 640
    assert findings["input.height"]["actual"] == 320
    assert "      height: 640\n" in human
    assert "      height: 320\n" in human
    assert "      FAIL\n" in human
    assert findings["semantic.color"]["status"] == "UNKNOWN"
    assert data["overall"] == "STRUCTURALLY_INCOMPATIBLE"
    assert data["exit_code"] == 1


def test_runtime_error_not_evaluated_layers_agree() -> None:
    report = runtime_model_error_report()
    data = _assert_substantive_equivalence(report)
    human = render_human(report)
    assert data["layers"]["frigate_structural"] == "NOT_EVALUATED"
    assert data["layers"]["semantic_uncertainty"] == "NOT_EVALUATED"
    assert data["model_metadata"] is None
    assert "Frigate Structural Contract\n  status: NOT_EVALUATED\n" in human
    assert "Semantic / Unproven Properties\n  status: NOT_EVALUATED\n" in human
    assert "Color channel meaning" not in human
    assert data["overall"] == "RUNTIME_MODEL_ERROR"
    assert data["exit_code"] == 2


def test_unsupported_after_inspection_metadata_and_overall_agree() -> None:
    report = unsupported_report()
    data = _assert_substantive_equivalence(report)
    human = render_human(report)
    assert data["model_metadata"] is not None
    assert data["model_metadata"]["input_count"] == 1
    assert data["model_metadata"]["output_count"] == 2
    assert "Inputs: 1\n" in human
    assert data["layers"]["semantic_uncertainty"] == "NOT_EVALUATED"
    assert "Semantic / Unproven Properties\n  status: NOT_EVALUATED\n" in human
    assert not any(
        item["layer"] == "frigate_structural"
        and item["status"] in {"PASS", "UNKNOWN"}
        and item["check_id"] != "operation.failure"
        for item in data["findings"]
    )
    assert data["overall"] == "INDETERMINATE_UNSUPPORTED"
    assert data["exit_code"] == 4


def test_explicit_runtime_identity_fields_agree() -> None:
    report = nine_output_explicit_runtime_report()
    data = _assert_substantive_equivalence(report)
    human = render_human(report)
    runtime = data["runtime"]
    assert runtime == EXPLICIT_RUNTIME
    assert runtime["selection_mode"] == "explicit"
    assert runtime["identity_verification"] == "PASS"
    for key, label in RUNTIME_HUMAN_LABELS.items():
        assert f"  {label}: {runtime[key]}\n" in human
    default_data = _assert_substantive_equivalence(structurally_compatible_report())
    assert default_data["runtime"] == DEFAULT_RUNTIME
    assert "requested_directory" not in default_data["runtime"]
    assert "identity_verification" not in default_data["runtime"]


def test_nine_output_metadata_and_structural_failures_agree() -> None:
    report = nine_output_explicit_runtime_report()
    data = _assert_substantive_equivalence(report)
    human = render_human(report)
    meta = data["model_metadata"]
    assert meta["input_count"] == 1
    assert meta["output_count"] == 9
    assert [item["index"] for item in meta["outputs"]] == list(range(9))
    assert "Outputs: 9\n" in human
    findings = {item["check_id"]: item for item in data["findings"]}
    assert findings["output.index_range"]["status"] == "FAIL"
    assert findings["output.0.reshape"]["status"] == "FAIL"
    assert findings["output.8.reshape"]["status"] == "FAIL"
    assert findings["input.count"]["status"] == "PASS"
    assert findings["semantic.color"]["status"] == "UNKNOWN"
    assert "  Output index range\n" in human
    assert "  Output 0 reshape\n" in human
    assert "  Output 8 reshape\n" in human
    assert "  Input count\n" in human
    assert "  Color channel meaning\n" in human


def test_model_path_authority_is_top_level_only() -> None:
    report = nine_output_explicit_runtime_report()
    data = _assert_substantive_equivalence(report)
    assert data["model_path"] == "/models/nine-output.rknn"
    assert "model_path" not in data["model_metadata"]
    human = render_human(report)
    assert "  inspected path: /models/nine-output.rknn\n" in human
    assert "probe-internal.rknn" not in human
    assert "probe-internal.rknn" not in render_json(report)


def test_schema_version_one_and_additive_metadata_preserve_legacy_keys() -> None:
    report = structurally_compatible_report()
    data = _assert_substantive_equivalence(report)
    assert data["schema_version"] == 1
    assert set(LEGACY_JSON_KEYS).issubset(data)
    assert set(data) == LEGACY_JSON_KEYS | {"model_metadata"}
    for finding in data["findings"]:
        assert {
            "check_id", "layer", "handler", "severity", "status", "summary", "expected", "actual", "evidence",
        }.issubset(finding)


def test_invalid_input_and_internal_error_overall_equivalence() -> None:
    invalid = Report(
        Overall.INVALID_INPUT,
        (_finding("operation.failure", Layer.FRIGATE_STRUCTURAL, Status.FAIL, "bad config"),),
    )
    internal = Report(
        Overall.INTERNAL_ERROR,
        (_finding("operation.failure", Layer.RUNTIME_MODEL, Status.FAIL, "unexpected internal failure"),),
    )
    for report, overall, exit_code in (
        (invalid, "INVALID_INPUT", 3),
        (internal, "INTERNAL_ERROR", 5),
    ):
        data = _assert_substantive_equivalence(report)
        assert data["overall"] == overall
        assert data["exit_code"] == exit_code
        assert data["model_metadata"] is None
        assert data["layers"]["semantic_uncertainty"] == "NOT_EVALUATED"


def test_renderers_are_deterministic_for_shared_report() -> None:
    report = nine_output_explicit_runtime_report()
    humans = [render_human(report) for _ in range(3)]
    machines = [render_json(report) for _ in range(3)]
    assert len(set(humans)) == 1
    assert len(set(machines)) == 1
    assert json.loads(machines[0])["schema_version"] == 1
