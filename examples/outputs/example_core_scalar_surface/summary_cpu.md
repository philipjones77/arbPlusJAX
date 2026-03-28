# Example Core Scalar Surface Summary (cpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `cpu`
- devices: `['TFRT_CPU_0']`
- benchmark_rows: `15`
- profile_rows: `288`
- comparison_rows: `2`
- batch_padding_rows: `10`

## Benchmark Operations

- `acf_mul`
- `arb_fpwrap_double_exp`
- `arf_add`
- `fmpr_mul`
- `fmpzi_add`

## Backend Summary

- `scipy`: mean_time_ms=0.00457108, mean_containment=1
- `c_arb`: mean_time_ms=0.11318, mean_containment=nan
- `jax_point`: mean_time_ms=0.176407, mean_containment=1
- `jax_adaptive`: mean_time_ms=0.241981, mean_containment=0
- `jax_rigorous`: mean_time_ms=0.243591, mean_containment=0
- `jax_basic`: mean_time_ms=0.311699, mean_containment=0
- `mpmath`: mean_time_ms=0.927102, mean_containment=1
- `boost`: mean_time_ms=5.72188, mean_containment=1

## Batch Padding Speed

- `arf_add` / `api_batch_unpadded`: warm=0.000237522s, recompile=0.0232654s
- `arf_add` / `api_batch_padded`: warm=1.65327e-05s, recompile=0.0259137s
- `acf_mul` / `api_batch_unpadded`: warm=2.69357e-05s, recompile=0.0287201s
- `acf_mul` / `api_batch_padded`: warm=2.15477e-05s, recompile=0.0268203s
- `fmpr_mul` / `api_batch_unpadded`: warm=5.97483e-05s, recompile=0.0283798s
- `fmpr_mul` / `api_batch_padded`: warm=2.09983e-05s, recompile=0.0236634s
- `fmpzi_add` / `api_batch_unpadded`: warm=7.0604e-05s, recompile=0.0526233s
- `fmpzi_add` / `api_batch_padded`: warm=6.5187e-05s, recompile=0.0411863s
- `arb_fpwrap_double_exp` / `api_batch_unpadded`: warm=2.1468e-05s, recompile=0.0280552s
- `arb_fpwrap_double_exp` / `api_batch_padded`: warm=1.82663e-05s, recompile=0.0311244s
