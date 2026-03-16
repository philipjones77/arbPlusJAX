Last updated: 2026-03-14T00:00:00Z

# Tests

`tests/` is the canonical home for correctness checks. It is separate from `benchmarks/`, which is the canonical home for performance and cross-backend comparison runs.

## Test families

- `test_*_chassis.py`: primary implementation tests for the JAX runtime surface.
  - this now includes sparse matrix chassis coverage in `test_srb_mat_chassis.py` and `test_scb_mat_chassis.py`
- `test_*_parity.py`: Arb/FLINT parity tests. These are gated behind `ARBPLUSJAX_RUN_PARITY=1`.
- `test_*_modes.py` and focused API tests: mode-wrapper, metadata, and contract checks.
- `test_all_functions_smoke.py`: broad import and entry-point smoke coverage.

## Harness entry point

Use `tools/run_test_harness.py` for environment-aware test runs.

Common profiles:

- `smoke`: fast import and surface smoke checks.
- `matrix`: dense / matrix-mode focused checks, including `arb_mat`, `acb_mat`, `jrb_mat`, `jcb_mat`, and `mat_modes`.
  - sparse chassis tests can be added to this profile or split into their own profile once the harness is updated
- `special`: tail acceleration and incomplete special-function stack.
- `chassis`: all non-parity tests under `tests/`.
- `parity`: parity-only tests under `tests/`.
- `bench-smoke`: benchmark CLI smoke tests under `benchmarks/`.
- `full`: chassis first, then optional benchmark smoke and parity.

Use `--outdir` when you want the harness to write a shared environment header:

```bash
python tools/run_test_harness.py --profile chassis --jax-mode cpu --outdir results/test_runs/chassis_cpu
```

This writes `runtime_manifest.json`, which uses the same schema as benchmark runs.

Examples:

Linux or macOS:

```bash
python tools/run_test_harness.py --profile chassis --jax-mode cpu
python tools/run_test_harness.py --profile matrix --jax-mode cpu
ARB_C_REF_DIR="$PWD/stuff/migration/c_chassis/build" \
LD_LIBRARY_PATH="$ARB_C_REF_DIR:$LD_LIBRARY_PATH" \
python tools/run_test_harness.py --profile parity --jax-mode cpu
```

Windows PowerShell:

```powershell
python .\tools\run_test_harness.py --profile chassis --jax-mode cpu
python .\tools\run_test_harness.py --profile matrix --jax-mode cpu
$env:ARB_C_REF_DIR = "$PWD\stuff\migration\c_chassis\build"
python .\tools\run_test_harness.py --profile parity --jax-mode cpu
```

Google Colab:

```bash
!python /content/arbplusJAX/tools/run_test_harness.py --profile chassis --jax-mode gpu
!python /content/arbplusJAX/tools/run_test_harness.py --profile bench-smoke --jax-mode gpu
```

## Benchmark integration

- `benchmarks/test_benchmark_smoke.py` is intentionally small and only checks that the benchmark harness is invocable.
- Real benchmark sweeps are run through `tools/run_benchmarks.py` or `tools/run_harness_profile.py`.
- The test harness does not run full performance sweeps by default. That separation keeps correctness runs stable across Windows, Linux, and Colab.
