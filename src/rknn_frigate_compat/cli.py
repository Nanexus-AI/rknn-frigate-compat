"""Command-line orchestration for the RKNN Frigate compatibility checker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .checks import common_input_checks, multipart_checks, semantic_findings
from .config import ConfigError, load_config
from .models import ErrorSubtype, Finding, Layer, ModelMetadata, Overall, Report, Severity, Status
from .probe import (
    RUNTIME_ERROR_STAGES,
    ProbeError,
    ProbeFailure,
    ProbeLaunchError,
    ProtocolError,
    RuntimeDirectoryError,
    RuntimeSelection,
    RuntimeSelectionError,
    prepare_runtime,
    resolve_probe,
    run_probe,
    runtime_context,
)
from .report import render_human, render_json


class CliUsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(
        prog="rknn-frigate-compat",
        description="Check the mechanically provable part of an RKNN model's selected Frigate contract.",
        epilog="Results: 0 compatible-within-scope, 1 structural mismatch, 2 Runtime/model error, 3 invalid input, 4 unsupported, 5 internal error. No semantic or inference compatibility is proven.",
    )
    parser.add_argument("--model", required=True, type=Path, help="readable local .rknn model")
    parser.add_argument("--config", required=True, type=Path, help="complete Frigate YAML configuration")
    parser.add_argument("--detector", help="RKNN detector name when configuration is ambiguous")
    parser.add_argument("--probe", type=Path, help="explicit metadata probe executable")
    parser.add_argument("--runtime-lib-dir", type=Path, help="explicit directory containing librknnrt.so")
    parser.add_argument("--json", action="store_true", help="emit one schema-versioned JSON report")
    return parser


def _finding(check_id: str, layer: Layer, status: Status, summary: str, expected: object = None, actual: object = None) -> Finding:
    severity = Severity.ERROR if status is Status.FAIL else Severity.WARNING if status in {Status.WARN, Status.UNKNOWN} else Severity.INFO
    return Finding(check_id, layer, None, severity, status, summary, expected, actual, ())


def _failure(overall: Overall, message: str, *, layer: Layer = Layer.RUNTIME_MODEL, runtime: dict[str, object] | None = None,
             subtype: ErrorSubtype | None = None, selected_detector: str | None = None, handler: str | None = None,
             model_path: str | None = None, configured_model_path: str | None = None,
             model_metadata: ModelMetadata | None = None,
             completed_findings: tuple[Finding, ...] = ()) -> Report:
    return Report(
        overall=overall,
        findings=completed_findings + (_finding("operation.failure", layer, Status.FAIL, message),),
        selected_detector=selected_detector,
        handler=handler,
        model_path=model_path,
        configured_model_path=configured_model_path,
        runtime=runtime or {},
        error_subtype=subtype,
        model_metadata=model_metadata,
    )


def _usable_file(path: Path, name: str) -> None:
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ConfigError(f"{name} must be a readable regular file: {path}")


def _runtime_pass() -> Finding:
    return _finding("runtime.identity", Layer.RUNTIME_MODEL, Status.PASS, "Loaded Runtime identity was independently verified")


def evaluate(args: argparse.Namespace) -> Report:
    _usable_file(args.model, "model")
    _usable_file(args.config, "config")
    config = load_config(args.config, args.detector)
    selection = prepare_runtime(args.runtime_lib_dir)
    try:
        probe_path = resolve_probe(args.probe)
        result = run_probe(probe_path, args.model, selection)
    except RuntimeSelectionError as exc:
        return _failure(
            Overall.RUNTIME_MODEL_ERROR, str(exc),
            runtime=runtime_context(selection, exc.identity, verified=False),
            subtype=ErrorSubtype.RUNTIME_SELECTION_ERROR,
            selected_detector=config.detector, handler=config.model_type,
            model_path=str(args.model), configured_model_path=config.configured_model_path,
        )
    if isinstance(result, ProbeError):
        overall = Overall.RUNTIME_MODEL_ERROR if result.stage in RUNTIME_ERROR_STAGES else Overall.INTERNAL_ERROR
        completed = (_runtime_pass(),) if result.runtime is not None else ()
        return _failure(
            overall, f"probe {result.stage}: {result.code}: {result.message}",
            runtime=runtime_context(selection, result.runtime, verified=result.runtime is not None),
            selected_detector=config.detector, handler=config.model_type,
            model_path=str(args.model), configured_model_path=config.configured_model_path,
            completed_findings=completed,
        )
    context = runtime_context(selection, result.runtime, verified=True)
    base = dict(
        selected_detector=config.detector,
        handler=config.model_type,
        model_path=str(args.model),
        configured_model_path=config.configured_model_path,
        runtime=context,
        model_metadata=result.model,
    )
    if config.model_path_kind != "custom":
        return _failure(Overall.INDETERMINATE_UNSUPPORTED, f"Frigate model path requires unsupported normalization: {config.model_path_kind}",
                        layer=Layer.FRIGATE_STRUCTURAL, completed_findings=(_runtime_pass(),), **base)
    if config.model_type != "yolo-generic":
        return _failure(Overall.INDETERMINATE_UNSUPPORTED, f"unsupported handler: {config.model_type}",
                        layer=Layer.FRIGATE_STRUCTURAL, completed_findings=(_runtime_pass(),), **base)
    if result.model.output_count <= 1:
        return _failure(Overall.INDETERMINATE_UNSUPPORTED, "yolo-generic single/zero-output branch is unsupported",
                        layer=Layer.FRIGATE_STRUCTURAL, completed_findings=(_runtime_pass(),), **base)
    findings = [_runtime_pass()] + common_input_checks(config, result.model) + multipart_checks(result.model) + semantic_findings(config)
    overall = Overall.STRUCTURALLY_INCOMPATIBLE if any(item.status is Status.FAIL for item in findings) else Overall.STRUCTURALLY_COMPATIBLE
    return Report(overall=overall, findings=tuple(findings), **base)


def main(argv: list[str] | None = None) -> int:
    actual = sys.argv[1:] if argv is None else argv
    json_requested = "--json" in actual
    try:
        args = build_parser().parse_args(actual)
        report = evaluate(args)
    except (CliUsageError, ConfigError, RuntimeDirectoryError) as exc:
        report = _failure(Overall.INVALID_INPUT, str(exc), layer=Layer.FRIGATE_STRUCTURAL)
    except (ProbeLaunchError, ProtocolError, ProbeFailure) as exc:
        report = _failure(Overall.INTERNAL_ERROR, str(exc))
    except Exception as exc:  # deterministic public boundary; never expose traceback
        report = _failure(Overall.INTERNAL_ERROR, f"unexpected internal failure: {exc}")
    sys.stdout.write(render_json(report) + "\n" if json_requested else render_human(report))
    return report.exit_code
