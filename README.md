# rknn-frigate-compat

Have an RKNN model for RK3588 that looks valid but does not behave as expected
in Frigate? An RKNN model may convert successfully, load with the Rockchip
Runtime, and still not match the structural contract expected by Frigate. This
tool checks the model's real Runtime metadata against the selected supported
Frigate contract before you spend time debugging a full deployment.

## Why this tool exists

Frigate community discussions have highlighted this source of confusion: a
`.rknn` file can be valid for Rockchip Runtime while still exposing output
shapes, counts, or layouts that are inconsistent with the selected Frigate
handler. This project turns that gap into a conservative, local structural
check.

This project is independent of Frigate and is not part of the Frigate
repository. It does not redistribute Frigate source. The compatibility rules in
this release were derived from Frigate's RKNN / `yolo-generic` behavior as of
September 2026. Future Frigate changes may require corresponding contract
updates.

## What it checks

This tool answers:

> Can the observable RKNN structure be mechanically proven compatible with a
> supported Frigate RKNN handler contract?

It compares Runtime-reported tensor metadata (counts, shapes, layouts, dtypes,
and quantization metadata) with the mechanically defined portion of the
selected supported Frigate contract.

## What it does not prove

It does **not** answer:

> Will this model produce correct detections?

Structural compatibility is not inference correctness. The checker does not run
inference, validate labels, prove preprocessing, or certify a production
Frigate deployment. Properties that cannot be proven from available metadata
remain `UNKNOWN`. See [docs/report.md](docs/report.md).

## Supported scope

Public version `0.2.0` currently supports this path only:

- detector: `rknn`
- `model_type: yolo-generic`
- multipart output path (more than one output)
- custom model path (slash-containing Frigate `model.path`)

It does not claim support for arbitrary Frigate or RKNN models. Unsupported
examples and the full boundary are documented in
[docs/compatibility.md](docs/compatibility.md).

## Quick start

1. Install the Python CLI from this source tree (Python 3.10+, PyYAML 6.x).
2. Prepare a Rockchip RKNN Runtime/header environment (`rknn_api.h`,
   `librknnrt.so`) obtained separately.
3. Build the native metadata probe once with
   `tools/build_rknn_metadata_probe.py`.
4. Run the checker with a local `.rknn` model and a complete Frigate YAML
   config.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .

python tools/build_rknn_metadata_probe.py \
  --include-dir /path/to/rknn/include \
  --library-dir /path/to/rknn/lib \
  --prefix "$VIRTUAL_ENV"

rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml
```

Build the probe into the same environment where the CLI is installed. With the
venv activated, `--prefix "$VIRTUAL_ENV"` matches the path the CLI uses to find
`libexec/rknn-frigate-compat/rknn-metadata-probe`.

Detailed installation, Runtime selection, and troubleshooting are in
[docs/usage.md](docs/usage.md).

## RKNN Runtime and metadata probe

The Python CLI itself does not require native compilation. The RKNN metadata
probe is a separate small C executable built locally against a user-supplied
Rockchip `rknn_api.h` and `librknnrt.so`.

This repository does **not** redistribute:

- `librknnrt.so`
- Rockchip `rknn_api.h`
- RKNN Toolkit / SDK
- model-zoo binaries
- `.rknn` models

Users obtain those assets separately and are responsible for Rockchip licensing
terms. This project is not affiliated with Rockchip. See
[docs/usage.md](docs/usage.md).

## Understanding the result

| Result | Meaning |
|---|---|
| `PASS` | A specific mechanically checked condition matched |
| `FAIL` | A mechanically proven mismatch against the selected supported contract |
| `UNKNOWN` | Considered, but available metadata cannot prove it |

Overall outcomes include `STRUCTURALLY_COMPATIBLE` (exit 0) and
`STRUCTURALLY_INCOMPATIBLE` (exit 1). Exit 0 is structural compatibility within
the supported mechanically checked scope only — not semantic or inference
compatibility. Full report semantics are in [docs/report.md](docs/report.md).

## Extensibility

The RKNN metadata inspection layer is more general than the currently
implemented Frigate compatibility contract. The native probe can expose generic
Runtime metadata such as input/output count, shape, layout, dtype, and
quantization metadata. It is not intrinsically tied to one YOLO family.

The current limitation is the set of Frigate contracts implemented by the
checker. Future releases may add additional Frigate RKNN handler contracts as
Frigate adds or changes supported RKNN model paths, without redesigning the
underlying metadata probe. No specific future handlers or dates are promised.
See [docs/compatibility.md](docs/compatibility.md).

## Real RK3588 validation

One real-world RK3588 / aarch64 validation example is documented under
[docs/validation/rk3588-validation.md](docs/validation/rk3588-validation.md).
That example measured a nine-output model as
`STRUCTURALLY_INCOMPATIBLE` (exit 1) against the supported `yolo-generic`
multipart contract, with semantic properties remaining `UNKNOWN`.

Sanitized captures:

- [docs/examples/real-human-report.txt](docs/examples/real-human-report.txt)
- [docs/examples/real-json-report.json](docs/examples/real-json-report.json)

This is one validated example, not a claim about universal RK3588 behavior.

## Project scope and contributions

This is intentionally a small, focused utility rather than a framework intended
to grow indefinitely. Users are welcome to use, fork, modify, and adapt it. Bug
reports and concrete compatibility findings may be useful.

Active external feature contribution is not required. Longer term, useful
compatibility ideas may be better upstreamed into the broader Frigate ecosystem
than expanded here as a large standalone framework. That is a direction, not an
acceptance promise.

## License

MIT. See [LICENSE](LICENSE).
