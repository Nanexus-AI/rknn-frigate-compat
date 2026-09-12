# Compatibility boundary

This document defines what public `rknn-frigate-compat` 0.2.0 can and cannot
conclude. Project overview remains in the [README](../README.md).

## Independence and references

This project is independent of Frigate and is not part of the Frigate
repository. It does not redistribute Frigate source. The compatibility rules in
this release were derived from Frigate's RKNN / `yolo-generic` behavior as of
September 2026. That contract reflects a specific point in Frigate development
and may need updating as Frigate changes.

The summaries below are independent descriptions of this checker's implemented
scope. They are not copied Frigate source text or tables.

This project is also not affiliated with Rockchip. Rockchip Runtime/header/SDK
and model assets are external and are not redistributed here.

## Architecture

```text
generic RKNN metadata layer
  +
explicit Frigate contract layer
```

1. **Generic RKNN metadata layer** — the native probe inspects Runtime-reported
   input/output counts, shapes, layouts, dtypes, and quantization metadata. That
   inspection is not intrinsically tied to one YOLO family.
2. **Explicit Frigate contract layer** — the checker applies only mechanically
   defined checks for Frigate handler contracts it implements.

The current limitation is the set of Frigate contracts implemented by the
checker, not the breadth of metadata the probe can observe.

## Currently supported path

Public version `0.2.0` supports this combination:

| Requirement | Value |
|---|---|
| Detector type | `rknn` |
| `model_type` | `yolo-generic` |
| Output branch | multipart (more than one output) |
| Frigate `model.path` | custom path containing `/` |

Within that path, the checker mechanically evaluates common input structure and
multipart output index/rank/layout/reshape conditions against Runtime metadata.
See [report.md](report.md) for Finding and overall semantics.

## Current exclusions

Examples outside the implemented mechanical scope include:

- Frigate model presets (paths that are not slash-containing custom paths)
- `plus://` model paths
- non-`yolo-generic` handlers such as defaults that resolve to other families
  (for example SSD-oriented defaults), YOLO-NAS, YOLOX, and similar paths
- `yolo-generic` single-output or zero-output branches
- non-`rknn` detectors

Those selections produce `INDETERMINATE_UNSUPPORTED` (exit 4). That result
means the selected Frigate path is outside this checker's implemented scope.

## Unsupported is not incompatible

```text
unsupported != incompatible
```

`INDETERMINATE_UNSUPPORTED` is not a claim that the model is broken, that
Rockchip Runtime rejects it, or that Frigate can never use it under another
handler or future contract. It only means this tool did not mechanically prove
compatibility or incompatibility for that selection.

Likewise, `STRUCTURALLY_INCOMPATIBLE` is a mismatch against the selected
supported contract only. It is not a universal verdict about every Frigate
handler or every deployment.

## Extensibility

New Frigate RKNN handler contracts can be supported by adding explicit,
mechanically defined checks while reusing the same metadata inspection layer.
As Frigate adds or changes supported RKNN model paths, future releases of this
tool may grow the contract set without redesigning the probe.

No specific future handlers or release dates are promised.

## Related docs

- [usage.md](usage.md)
- [report.md](report.md)
- [validation/rk3588-validation.md](validation/rk3588-validation.md)
