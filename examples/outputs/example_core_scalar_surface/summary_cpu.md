# Example Core Scalar Surface Summary (cpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `cpu`
- devices: `['TFRT_CPU_0']`
- benchmark_rows: `5`
- profile_rows: `288`
- comparison_rows: `2`

## Benchmark Operations

- `acf_mul`
- `arb_fpwrap_double_exp`
- `arf_add`
- `fmpr_mul`
- `fmpzi_add`

## Backend Summary

- `scipy`: mean_time_ms=0.00823333, mean_containment=1
- `jax_point`: mean_time_ms=0.269623, mean_containment=1
- `jax_rigorous`: mean_time_ms=0.33923, mean_containment=0
- `jax_adaptive`: mean_time_ms=0.360795, mean_containment=0
- `c_arb`: mean_time_ms=0.374501, mean_containment=nan
- `jax_basic`: mean_time_ms=0.392936, mean_containment=0
- `mpmath`: mean_time_ms=1.34525, mean_containment=1
- `boost`: mean_time_ms=9.7323, mean_containment=1
