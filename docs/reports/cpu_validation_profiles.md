Last updated: 2026-03-24T00:00:00Z

# CPU Validation Profiles

This report records the latest bounded CPU validation profiles run through
[tools/run_test_harness.py](/tools/run_test_harness.py).

Policy lives in:

- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)

## Current CPU Validation Slice

Run date:

- 2026-03-23
- 2026-03-24

Interpreter and runtime:

- Python: `/home/phili/miniforge3/envs/jax/bin/python`
- requested JAX mode: `cpu`
- realized backend: `cpu`
- device set: `TFRT_CPU_0`
- `jax_enable_x64`: `false`
- environment kind: `wsl`

## Commands Run

- `python tools/run_test_harness.py --profile matrix --jax-mode cpu --outdir tests/_runs/matrix_cpu_2026-03-23`
- `python tools/run_test_harness.py --profile special --jax-mode cpu --outdir tests/_runs/special_cpu_2026-03-23`
- `python tools/run_test_harness.py --profile bench-smoke --jax-mode cpu --outdir tests/_runs/bench_smoke_cpu_2026-03-23`
- `python tools/run_test_harness.py --profile sparse --jax-mode cpu --outdir tests/_runs/sparse_cpu_2026-03-24`

## Results

- `matrix`: `114 passed`, `48 warnings`, `289.72s`
- `special`: `64 passed`, `39.43s`
- `bench-smoke`: `7 passed`, `15 deselected`, `16.08s`
- `sparse`: `48 passed`, `196.60s`

## What This Confirms

- the explicit CPU harness path remains healthy for the dense and matrix-free
  chassis slice
- the current incomplete-tail and special-function tranche remains green under
  the dedicated `special` profile
- the sparse/block/vblock point/basic slice now has a retained CPU harness run
  instead of only a manifest stub
- benchmark smoke remains separated from correctness ownership while still
  exercising the normalized benchmark CLI surface on CPU
- the runtime-manifest path works consistently across these test profiles

## Current Limits

- this is a bounded CPU validation slice, not the full `--profile chassis`
  run
- GPU validation remains a later dedicated tranche
- parity remains opt-in and separate from these CPU harness runs
