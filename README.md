# arbPlusJAX

Implementation of arb for JAX.

## Layout

- `src/arbplusjax/`: JAX implementation (primary runtime).
- `tests/`: chassis + parity tests.
- `benchmarks/`: benchmark and cross-backend comparison scripts.
- `tools/`: utility scripts (packaging, benchmark runner/report tools).
- `docs/`: theory, implementation, practical, and governance notes.
- `contracts/`: binding runtime/API guarantees.
- `examples/`: usage notebooks and experiments.
- `experiments/`: exploratory work.
- `outputs/`: canonical output root going forward.
- `results/`: legacy benchmark output location retained for compatibility.

## Run Platform

This repo remains the active development workspace and also serves as a repeatable run platform for Windows, Linux, and Google Colab.

- `tests/` is the correctness surface.
- `benchmarks/` is the performance and comparison surface.
- `tools/run_test_harness.py` is the dedicated test harness.
- `tools/run_benchmarks.py` and `tools/run_harness_profile.py` are the benchmark runners.

See [docs/practical/README.md](/home/phili/projects/arbplusJAX/docs/practical/README.md), [docs/implementation/run_platform.md](/home/phili/projects/arbplusJAX/docs/implementation/run_platform.md), and [tests/README.md](/home/phili/projects/arbplusJAX/tests/README.md) for the exact split.

## VS Code Workspace

The VS Code root should be the repo root, not `docs/`.

- [arbPlusJAX.code-workspace](/home/phili/projects/arbplusJAX/arbPlusJAX.code-workspace): generic repo-root workspace
- [arbPlusJAX-linux.code-workspace](/home/phili/projects/arbplusJAX/arbPlusJAX-linux.code-workspace): Linux-focused interpreter and terminal defaults
- [arbPlusJAX-windows.code-workspace](/home/phili/projects/arbplusJAX/arbPlusJAX-windows.code-workspace): Windows-focused interpreter and terminal defaults

Use the Linux workspace on Linux and the Windows workspace on Windows. Both keep the explorer rooted at the repository root.

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

Use the dedicated test harness when possible.

Chassis tests (all OS):

```bash
python tools/run_test_harness.py --profile chassis --jax-mode cpu
```

Parity tests (requires Arb C libs):

Windows PowerShell:
```powershell
$env:ARB_C_REF_DIR = "C:\path\to\arbPlusJAX\stuff\migration\c_chassis\build"
python .\tools\run_test_harness.py --profile parity --jax-mode cpu
```

Linux/macOS (bash/zsh):
```bash
export ARB_C_REF_DIR=/path/to/arbPlusJAX/stuff/migration/c_chassis/build
export LD_LIBRARY_PATH="$ARB_C_REF_DIR:$LD_LIBRARY_PATH"
python tools/run_test_harness.py --profile parity --jax-mode cpu
```

Optional benchmark smoke tests (included in standard pytest discovery):

Windows PowerShell:
```powershell
python .\tools\run_test_harness.py --profile bench-smoke --jax-mode cpu
```

Linux/macOS (bash/zsh):
```bash
python tools/run_test_harness.py --profile bench-smoke --jax-mode cpu
```

## Parity tools

The scripts in `tools/` load Arb reference libraries built in the Arb workspace or archived under `stuff/`.

## Notes

- JAX is the primary implementation. The Arb workspace is used for parity and benchmarking.
- Some advanced helper routines are approximations; see `docs/` for details.
