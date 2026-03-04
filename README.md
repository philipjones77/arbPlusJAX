# arbPlusJAX

Implementation of arb for JAX.

## Layout

- `src/arbplusjax/`: JAX implementation (primary runtime).
- `tests/`: chassis + parity tests.
- `benchmarks/`: benchmark and cross-backend comparison scripts.
- `tools/`: utility scripts (packaging, benchmark runner/report tools).
- `docs/`: theory and implementation notes.
- `examples/`, `notebooks/`: usage and experiments.
- `results/`: benchmark outputs (what ran and when).

## Install (editable)

Windows PowerShell:
```powershell
python -m pip install -e .
```

Linux/macOS (bash/zsh):
```bash
python -m pip install -e .
```

## Tests

Chassis tests (all OS):

```bash
python -m pytest tests -q -m "not parity"
```

Parity tests (requires Arb C libs):

Windows PowerShell:
```powershell
$env:ARB_C_REF_DIR = "C:\path\to\arbPlusJAX\stuff\migration\c_chassis\build"
$env:ARBPLUSJAX_RUN_PARITY = "1"
python -m pytest tests -q -m parity
```

Linux/macOS (bash/zsh):
```bash
export ARB_C_REF_DIR=/path/to/arbPlusJAX/stuff/migration/c_chassis/build
export LD_LIBRARY_PATH="$ARB_C_REF_DIR:$LD_LIBRARY_PATH"
export ARBPLUSJAX_RUN_PARITY=1
python -m pytest tests -q -m parity
```

Optional benchmark smoke tests (included in standard pytest discovery):

Windows PowerShell:
```powershell
$env:ARBPLUSJAX_RUN_BENCHMARKS = "1"
python -m pytest benchmarks -q -m benchmark
```

Linux/macOS (bash/zsh):
```bash
export ARBPLUSJAX_RUN_BENCHMARKS=1
python -m pytest benchmarks -q -m benchmark
```

## Parity tools

The scripts in `tools/` load Arb reference libraries built in the Arb workspace or archived under `stuff/`.

## Notes

- JAX is the primary implementation. The Arb workspace is used for parity and benchmarking.
- Some advanced helper routines are approximations; see `docs/` for details.
