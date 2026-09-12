# Report semantics

The checker emits one authoritative report in human text (default) or JSON
(`--json`). Both forms share the same findings, overall result, and exit code.

Sanitized real RK3588 captures:

- [examples/real-human-report.txt](examples/real-human-report.txt)
- [examples/real-json-report.json](examples/real-json-report.json)

## Five sections

### 1. Selection / Environment

Identifies:

- selected detector
- configured handler (`model_type`)
- evaluated contract branch when available (for example `yolo-generic:multipart`)
- configured Frigate model path versus inspected CLI model path
- Runtime selection mode, loader identity, API/driver versions when available

### 2. RKNN Model Metadata

Observational Runtime metadata only:

- input/output counts
- per-tensor shape, layout, dtype, quantization

Metadata does not by itself identify a model family, prove Frigate
compatibility, or establish tensor meaning.

### 3. Frigate Structural Contract

Mechanically checked conditions for the selected supported contract. Each
completed check is rendered from one checker-owned Finding:

| Label | Authority |
|---|---|
| `Frigate expects` | Finding expected value |
| `RKNN model reports` | Finding actual value, with tensor context when useful |
| `Result` | Unchanged Finding status and reason |

The renderer formats values; it does not recalculate compatibility.

### 4. Semantic / Unproven Properties

Properties considered but not mechanically proven from available metadata.
Examples include RGB/BGR meaning, preprocessing/normalization, DFL
representation, objectness and class-score semantics, box/anchor/label meaning,
and RKNN Lite numerical/dequantization behavior.

### 5. Overall

Authoritative overall result and public exit code, plus a short summary line.

## Finding statuses

| Status | Meaning |
|---|---|
| `PASS` | That specific mechanically checked condition was satisfied |
| `FAIL` | A mechanically proven mismatch exists against the selected supported contract |
| `UNKNOWN` | The property was considered, but available metadata cannot prove it |
| `NOT_EVALUATED` | That stage did not run (for example after Runtime/model inspection failure) |

`PASS` does not prove semantic compatibility, successful inference, correct
labels, preprocessing, numerical correctness, or production integration.

`FAIL` does not mean the RKNN model is broken, cannot work elsewhere, or is
incompatible with every Frigate handler.

`UNKNOWN` is neither failure nor a probability claim that the property is
likely compatible or incompatible.

## Overall outcomes and exit codes

| Exit | Overall | Meaning |
|---:|---|---|
| 0 | `STRUCTURALLY_COMPATIBLE` | No mechanically proven structural mismatch within the selected supported contract |
| 1 | `STRUCTURALLY_INCOMPATIBLE` | At least one mechanically proven structural mismatch |
| 2 | `RUNTIME_MODEL_ERROR` | Runtime selection or model inspection failed |
| 3 | `INVALID_INPUT` | Invalid input or configuration |
| 4 | `INDETERMINATE_UNSUPPORTED` | Selected Frigate path is outside implemented mechanical scope |
| 5 | `INTERNAL_ERROR` | Internal failure prevented complete evaluation |

Critical distinction:

```text
STRUCTURALLY_COMPATIBLE != inference correctness
```

Exit 0 is structural compatibility within the supported mechanically checked
scope only.

## Concise examples

Compatible-within-scope overall:

```text
Overall
  STRUCTURALLY_COMPATIBLE (exit 0)
  Compatible within the mechanically checked structural scope; does not prove
  semantic compatibility or production Frigate integration.
```

Structural mismatch overall:

```text
Overall
  STRUCTURALLY_INCOMPATIBLE (exit 1)
  A mechanically proven structural mismatch was found against the selected
  supported contract.
```

A structural Finding with expected/actual authority:

```text
  Output index range
    Frigate expects
      supported indices: indices 0..2
    RKNN model reports
      output indices: [0, 1, 2, 3, 4, 5, 6, 7, 8]
      output count: 9
    Result
      FAIL
      Multipart output indices must fit the handler maps
```

## JSON report

`--json` emits one schema-version-1 document for the same authoritative Report.
It preserves Finding fields and values, includes Runtime identity fields, and
adds nullable `model_metadata` when inspection succeeded. It does not embed
human section prose.

## Related docs

- [usage.md](usage.md) — installation and CLI
- [compatibility.md](compatibility.md) — supported contract boundary
- [validation/rk3588-validation.md](validation/rk3588-validation.md) — one real example
