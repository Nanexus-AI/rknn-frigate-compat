"""Bounded extraction of compatibility fields from complete Frigate YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import FrigateConfig


DEFAULTS: dict[str, Any] = {
    "model_type": "ssd",
    "width": 320,
    "height": 320,
    "input_tensor": "nhwc",
    "input_pixel_format": "rgb",
    "input_dtype": "int",
}


class ConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def classify_model_path(value: str | None) -> str:
    if value is None:
        return "unsupported_missing"
    if value.startswith("plus://"):
        return "unsupported_plus"
    if "/" not in value and "\\" not in value:
        return "unsupported_preset"
    return "custom"


def load_config(path: Path, detector_name: str | None = None) -> FrigateConfig:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read Frigate configuration: {exc}") from None
    root = _mapping(raw, "configuration")
    detectors = _mapping(root.get("detectors"), "detectors")
    rknn: list[str] = []
    for name, value in detectors.items():
        detector = _mapping(value, f"detectors.{name}")
        detector_type = detector.get("type")
        if detector_type is not None and not isinstance(detector_type, str):
            raise ConfigError(f"detectors.{name}.type must be a string")
        if detector_type == "rknn":
            rknn.append(name)
    if detector_name is not None:
        if detector_name not in detectors:
            raise ConfigError(f"unknown detector: {detector_name}")
        selected = _mapping(detectors[detector_name], f"detectors.{detector_name}")
        if selected.get("type") != "rknn":
            raise ConfigError(f"selected detector is not RKNN: {detector_name}")
        selected_name = detector_name
    elif len(rknn) == 1:
        selected_name = rknn[0]
    elif not rknn:
        raise ConfigError("configuration contains no RKNN detector")
    else:
        raise ConfigError("multiple RKNN detectors require --detector: " + ", ".join(sorted(rknn)))

    model_value = root.get("model")
    model = {} if model_value is None else _mapping(model_value, "model")
    configured_path = model.get("path")
    if configured_path is not None:
        configured_path = _string(configured_path, "model.path")
    provenance: dict[str, str] = {}
    values: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        values[key] = model[key] if key in model else default
        provenance[key] = "explicit" if key in model else "default"
    values["model_type"] = _string(values["model_type"], "model.model_type").lower()
    values["width"] = _positive_int(values["width"], "model.width")
    values["height"] = _positive_int(values["height"], "model.height")
    values["input_tensor"] = _string(values["input_tensor"], "model.input_tensor").lower()
    values["input_pixel_format"] = _string(values["input_pixel_format"], "model.input_pixel_format").lower()
    values["input_dtype"] = _string(values["input_dtype"], "model.input_dtype").lower()
    if values["input_tensor"] not in {"nhwc", "nchw"}:
        raise ConfigError("model.input_tensor must be nhwc or nchw")
    if values["input_pixel_format"] not in {"rgb", "bgr", "yuv"}:
        raise ConfigError("model.input_pixel_format must be rgb, bgr, or yuv")
    if values["input_dtype"] not in {"int", "float"}:
        raise ConfigError("model.input_dtype must be int or float")
    return FrigateConfig(
        detector=selected_name,
        configured_model_path=configured_path,
        model_path_kind=classify_model_path(configured_path),
        model_type=values["model_type"],
        width=values["width"],
        height=values["height"],
        input_tensor=values["input_tensor"],
        input_pixel_format=values["input_pixel_format"],
        input_dtype=values["input_dtype"],
        provenance=provenance,
    )
