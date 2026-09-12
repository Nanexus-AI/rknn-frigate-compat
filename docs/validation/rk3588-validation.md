# RK3588 validation example

This page summarizes **one** real-world RK3588 validation example for
`rknn-frigate-compat` 0.2.0. It is not a claim about universal RK3588 behavior,
all Rockchip boards, or all `.rknn` models.

Sanitized report captures:

- [../examples/real-human-report.txt](../examples/real-human-report.txt)
- [../examples/real-json-report.json](../examples/real-json-report.json)

Paths below use public placeholders. After sanitization, the example files are
not byte-identical to internal evidence captures.

## Platform class

| Property | Observed example |
|---|---|
| Platform class | RK3588 / aarch64 |
| OS context | Ubuntu/Linux |
| RKNN API version | `2.3.2 (429f97ae6b@2025-04-09T09:09:27)` |
| Driver version | `0.9.6` |

## Model and Runtime identity

| Property | Observed example |
|---|---|
| Model basename | `c32_yolov8m_v2.0_0715_int8_300.rknn` |
| Model SHA-256 | `0ec5428c2d239a7bcfedebbdacdf577ad601cf024ca17a77b98206b30fe36b1e` |
| Runtime library | `librknnrt.so` (explicit selection) |
| Runtime SHA-256 | `d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8` |
| Identity verification | `PASS` |
| Frigate handler selection | detector `rknn`, `model_type: yolo-generic`, custom model path |

## Measured Runtime metadata

| Property | Observed example |
|---|---|
| Inputs | 1 |
| Input 0 | shape `[1, 640, 640, 3]`, layout `NHWC`, dtype `INT8`, quantization `AFFINE` |
| Outputs | 9 |
| Output shapes | `[1, 64, 80, 80]`, `[1, 32, 80, 80]`, `[1, 1, 80, 80]`, `[1, 64, 40, 40]`, `[1, 32, 40, 40]`, `[1, 1, 40, 40]`, `[1, 64, 20, 20]`, `[1, 32, 20, 20]`, `[1, 1, 20, 20]` (all `NCHW` / `INT8` / `AFFINE`) |

These values are Runtime observations only. They do not identify a model family
or prove inference correctness.

## Checker outcome

| Property | Observed example |
|---|---|
| Overall | `STRUCTURALLY_INCOMPATIBLE` |
| Exit code | `1` |
| Human/JSON agreement | same overall, exit, Runtime identity, metadata, and Finding statuses |
| Structural FAILs | output index range (expected indices `0..2`, actual nine outputs) and multipart reshape element checks for outputs `0..8` |
| Structural UNKNOWNs | input dtype conversion; input quantization/dequantization semantics |
| Semantic layer | all evaluated semantic properties `UNKNOWN` |

Interpretation:

- The model loaded and exposed metadata under the selected Rockchip Runtime.
- Against the supported Frigate `yolo-generic` multipart contract, the checker
  mechanically proved structural mismatches.
- Semantic and inference properties remained unproven (`UNKNOWN`).
- `STRUCTURALLY_INCOMPATIBLE` here is not a claim that the model is invalid for
  Rockchip Runtime or unusable under every other consumer.

## What this example does not show

- Universal RK3588 results for other models or Runtime builds
- Inference/detection correctness
- Production Frigate deployment readiness
- Support for handlers outside the current checker scope

For report field meanings, see [../report.md](../report.md). For the supported
contract boundary, see [../compatibility.md](../compatibility.md).
