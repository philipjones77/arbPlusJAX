# arbPlusJAX

Implementation of arb for JAX.

## Layout

- `src/arbplusjax/`: JAX implementation (primary runtime).
- `tests/`: chassis + parity tests.
- `benchmarks/`: benchmark and cross-backend comparison scripts.
- `tools/`: utility scripts (packaging, benchmark runner/report tools).
- `docs/`: theory and implementation notes.
- `examples/`: usage notebooks and experiments.
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

Run directly from source tree (no install):
```bash
PYTHONPATH=src python -m pytest tests -q -m "not parity"
```

## One-command Validation With Live Status

Linux/macOS (bash/zsh):
```bash
bash tools/run_validation.sh
```

Windows PowerShell:
```powershell
.\tools\run_validation.ps1
```

This prints timestamped progress and heartbeat lines so long test/benchmark phases do not look stuck.
By default it auto-detects and uses a `jax` conda env Python when present (for example `~/miniforge3/envs/jax/bin/python`).

Force interpreter and backend mode:
```bash
python tools/run_validation.py --python /home/phili/miniforge3/envs/jax/bin/python --jax-mode cpu
python tools/run_validation.py --python /home/phili/miniforge3/envs/jax/bin/python --jax-mode gpu
```

Windows PowerShell equivalents:
```powershell
python .\tools\run_validation.py --python C:\Users\phili\miniforge3\envs\jax\python.exe --jax-mode cpu
python .\tools\run_validation.py --python C:\Users\phili\miniforge3\envs\jax\python.exe --jax-mode gpu
```

Backend self-checks:
```bash
/home/phili/miniforge3/envs/jax/bin/python tools/check_jax_runtime.py --expect-backend cpu
JAX_PLATFORMS=cuda /home/phili/miniforge3/envs/jax/bin/python tools/check_jax_runtime.py --expect-backend gpu
```

For larger notebook sweeps with progress output:
```bash
python tools/run_notebook_sweeps.py --samples 2000,5000,10000 --seeds 0,1,2 --chunk-size 6 --jax-mode cpu
python tools/run_notebook_sweeps.py --samples 2000,5000,10000 --seeds 0,1,2 --chunk-size 6 --jax-mode gpu
```
Notebook template: `examples/example_large_sweeps_progress.ipynb`

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
