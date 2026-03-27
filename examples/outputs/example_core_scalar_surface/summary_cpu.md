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

- `scipy`: mean_time_ms=0.00953789, mean_containment=1
- `c_arb`: mean_time_ms=0.244852, mean_containment=nan
- `jax_adaptive`: mean_time_ms=0.495842, mean_containment=0
- `jax_rigorous`: mean_time_ms=0.522112, mean_containment=0
- `jax_basic`: mean_time_ms=0.570113, mean_containment=0
- `jax_point`: mean_time_ms=0.655237, mean_containment=1
- `mpmath`: mean_time_ms=2.15788, mean_containment=1
- `boost`: mean_time_ms=16.401, mean_containment=1

## Batch Padding Speed

- `arf_add` / `api_batch_unpadded`: warm=8.44303e-05s, recompile=0.0267424s
- `arf_add` / `api_batch_padded`: warm=2.23363e-05s, recompile=0.0414707s
- `acf_mul` / `api_batch_unpadded`: warm=3.30217e-05s, recompile=0.0281941s
- `acf_mul` / `api_batch_padded`: warm=0.000134104s, recompile=0.0352097s
- `fmpr_mul` / `api_batch_unpadded`: warm=3.0715e-05s, recompile=0.0251266s
- `fmpr_mul` / `api_batch_padded`: warm=5.133e-05s, recompile=0.0345944s
- `fmpzi_add` / `api_batch_unpadded`: warm=9.69727e-05s, recompile=0.0526591s
- `fmpzi_add` / `api_batch_padded`: warm=9.7787e-05s, recompile=0.0464118s
- `arb_fpwrap_double_exp` / `api_batch_unpadded`: warm=1.77327e-05s, recompile=0.0368558s
- `arb_fpwrap_double_exp` / `api_batch_padded`: warm=2.1767e-05s, recompile=0.0316696s
