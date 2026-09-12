# Usage

Detailed installation and usage for `rknn-frigate-compat` 0.2.0.
For project positioning and scope, start with the [README](../README.md).

## Python requirements

- Python 3.10 or newer
- PyYAML 6.x (installed automatically with the package)

Optional for Host tests:

```text
python3 -m pip install '.[test]'
```

## Source installation

From a clone of this repository:

```text
python3 -m venv .venv
.venv/bin/python -m pip install .
```

That installs the `rknn-frigate-compat` console entry point and the
`rknn_frigate_compat` Python package. The Python CLI does not require native
compilation.

A future pip/PyPI install path may be published later. This document does not
claim that the package is currently published on PyPI.

## Rockchip external prerequisites

Before building the metadata probe or inspecting a model, obtain separately:

| Asset | Role |
|---|---|
| `rknn_api.h` | Rockchip public C API header used at probe compile time |
| `librknnrt.so` | Rockchip RKNN Runtime shared library used at probe link/load time |

Typical sources include a Rockchip RKNN Toolkit / Runtime SDK layout or a
vendor board package. SDK directory layouts differ by release and vendor; this
tool does not assume one fixed tree.

This repository does **not** redistribute `rknn_api.h`, `librknnrt.so`, RKNN
Toolkit/SDK packages, model-zoo binaries, or `.rknn` models. Users are
responsible for Rockchip licensing terms. This project is not affiliated with
Rockchip and does not describe Rockchip SDK materials as open-source.

## Metadata probe architecture

```text
Python CLI (rknn-frigate-compat)
  → launches the local C probe as a subprocess
  → reads probe JSON metadata
  → compares metadata with the selected Frigate contract
```

The probe is a small C executable. It links against user-supplied Rockchip
header and Runtime paths, loads a `.rknn` model for metadata inspection, and
exits. It does not run Frigate inference or modify model/config files.

## Build the metadata probe

Use the repository helper exactly as implemented:

```text
python3 tools/build_rknn_metadata_probe.py \
  --include-dir /path/to/rknn/include \
  --library-dir /path/to/rknn/lib \
  --prefix /path/to/python/environment
```

Required arguments:

| Option | Meaning |
|---|---|
| `--include-dir` | Directory that contains `rknn_api.h` |
| `--library-dir` | Directory searchable for `-lrknnrt` (`librknnrt.so`) |
| `--prefix` | Install prefix; probe is written under `libexec/rknn-frigate-compat/` |

Optional:

| Option | Meaning |
|---|---|
| `--cc` | C compiler (default: `cc`) |

Example install location:

```text
/path/to/python/environment/libexec/rknn-frigate-compat/rknn-metadata-probe
```

The helper prints the installed probe path. It does not discover, download,
install, replace, or repair Runtime libraries.

If your Rockchip SDK uses a non-obvious layout, point `--include-dir` and
`--library-dir` at the directories that actually contain the header and
`librknnrt.so` for your aarch64 Linux RK3588 environment.

## Runtime library selection

### Default loader

```text
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml
```

Without `--runtime-lib-dir`, the probe uses the normal dynamic loader once. The
report records the actual loaded library path and SHA-256. No alternate Runtime
is scanned or retried.

### Explicit Runtime directory

```text
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml \
  --runtime-lib-dir /path/to/rknn/runtime
```

With `--runtime-lib-dir`, the requested directory, its canonical `librknnrt.so`
target, and the probe-reported loaded object must match by canonical path and
SHA-256 before metadata is trusted.

## Model and config invocation

```text
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml --json
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml \
  --detector rknn
rknn-frigate-compat --model /path/to/model.rknn --config /path/to/config.yml \
  --probe /path/to/rknn-metadata-probe
```

| Option | Meaning |
|---|---|
| `--model` | Readable local `.rknn` file to inspect |
| `--config` | Complete Frigate YAML configuration |
| `--detector` | RKNN detector name when the configuration is ambiguous |
| `--probe` | Explicit metadata probe executable |
| `--runtime-lib-dir` | Explicit directory containing `librknnrt.so` |
| `--json` | Emit one schema-versioned JSON report |

`--model` is the inspected host file. Frigate `model.path` describes Frigate's
configured selection and is not required to resolve to the same filesystem path.

## Exit codes

| Exit | Overall result |
|---:|---|
| 0 | `STRUCTURALLY_COMPATIBLE` |
| 1 | `STRUCTURALLY_INCOMPATIBLE` |
| 2 | `RUNTIME_MODEL_ERROR` |
| 3 | `INVALID_INPUT` |
| 4 | `INDETERMINATE_UNSUPPORTED` |
| 5 | `INTERNAL_ERROR` |

Exit 0 means structural compatibility within the supported mechanically checked
scope only. It does not prove correct detections. See
[report.md](report.md).

## Common setup mistakes

- Building the probe against headers that are not paired with the Runtime you
  later load.
- Pointing `--library-dir` at a directory that does not contain `librknnrt.so`
  for the target architecture.
- Using `--runtime-lib-dir` that does not contain a resolvable `librknnrt.so`.
- Expecting the tool to redistribute or auto-install Rockchip SDK assets.
- Treating `INDETERMINATE_UNSUPPORTED` as proof that a model is broken; that
  result means the selected Frigate path is outside the implemented checker
  scope. See [compatibility.md](compatibility.md).
- Interpreting `STRUCTURALLY_COMPATIBLE` as inference or detection validation.

## Runtime identity verification

When Runtime selection succeeds, the human and JSON reports expose fields such
as:

| Field | Use |
|---|---|
| `selection mode` / `selection_mode` | `default` or `explicit` |
| `requested directory` | Explicit mode only |
| `expected loader target` | Canonical `librknnrt.so` expected in explicit mode |
| `expected loader sha256` | Hash of the expected loader target |
| `loaded library path` | Canonical path the probe reported as loaded |
| `loaded library sha256` | Hash of the loaded library |
| `identity verification` | Explicit-mode path/hash agreement (`PASS` / `FAIL`) |
| `API version` | Runtime-reported API version string |
| `driver version` | Runtime-reported driver version string |

Use those fields to confirm which Runtime object actually supplied the metadata
before trusting structural conclusions.

## Host verification commands

```text
python3 -m pytest -q
python3 scripts/check_c_probe.py
python3 scripts/cli_smoke.py
python3 -m compileall -q src tests scripts tools
git diff --check
```

Host C tests use a stub/shim and are not RK3588 real-model validation.
