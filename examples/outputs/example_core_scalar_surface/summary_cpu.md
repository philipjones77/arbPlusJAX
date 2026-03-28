# Example Core Scalar Surface Summary (cpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `cpu`
- devices: `['TFRT_CPU_0']`
- benchmark_rows: `42`
- profile_rows: `288`
- comparison_rows: `2`
- batch_padding_rows: `37`

## Benchmark Operations

- `acf_mul`
- `arb_fpwrap_double_exp`
- `arf_add`
- `fmpr_mul`
- `fmpzi_add`

## Backend Summary

- `c_arb`: mean_time_ms=0.315926, mean_containment=nan
- `scipy`: mean_time_ms=0.317265, mean_containment=1
- `jax_rigorous`: mean_time_ms=0.609061, mean_containment=0
- `jax_point`: mean_time_ms=0.683174, mean_containment=1
- `jax_adaptive`: mean_time_ms=0.846823, mean_containment=0
- `jax_basic`: mean_time_ms=0.964298, mean_containment=0
- `mpmath`: mean_time_ms=2.69958, mean_containment=1
- `boost`: mean_time_ms=24.9921, mean_containment=1

## Batch Padding Speed

- `arf_add` / `api_batch_unpadded`: warm=3.04757e-05s, recompile=0.0316341s
- `arf_add` / `api_batch_padded`: warm=0.000138677s, recompile=0.0441043s
- `acf_mul` / `api_batch_unpadded`: warm=2.68173e-05s, recompile=0.0387624s
- `acf_mul` / `api_batch_padded`: warm=8.6314e-05s, recompile=0.0425516s
- `fmpr_mul` / `api_batch_unpadded`: warm=4.40207e-05s, recompile=0.039125s
- `fmpr_mul` / `api_batch_padded`: warm=0.000114325s, recompile=0.0515758s
- `fmpzi_add` / `api_batch_unpadded`: warm=6.18417e-05s, recompile=0.0654284s
- `fmpzi_add` / `api_batch_padded`: warm=0.00012678s, recompile=0.0685323s
- `arb_fpwrap_double_exp` / `api_batch_unpadded`: warm=1.8568e-05s, recompile=0.0502112s
- `arb_fpwrap_double_exp` / `api_batch_padded`: warm=1.763e-05s, recompile=0.039898s
- `arf_add` / `service_api_unpadded`: warm=0.000488062s, recompile=0.0394332s
- `arf_add` / `service_api_padded`: warm=0.00101421s, recompile=0.11758s
- `arf_add` / `service_api_bucketed`: warm=0.00177616s, recompile=0.121197s
- `fmpr_mul` / `service_api_unpadded`: warm=0.000358168s, recompile=0.0336498s
- `fmpr_mul` / `service_api_padded`: warm=0.000957073s, recompile=0.000884258s
- `fmpr_mul` / `service_api_bucketed`: warm=0.00102996s, recompile=0.0327796s
- `arb_fpwrap_double_exp` / `service_api_unpadded`: warm=0.00028641s, recompile=0.0427732s
- `arb_fpwrap_double_exp` / `service_api_padded`: warm=0.000570355s, recompile=0.000586453s
- `arb_fpwrap_double_exp` / `service_api_bucketed`: warm=0.000424152s, recompile=0.0364461s
- `acf_mul` / `service_api_unpadded`: warm=0.000508368s, recompile=0.0370512s
- `acf_mul` / `service_api_padded`: warm=0.00144545s, recompile=0.119227s
- `acf_mul` / `service_api_bucketed`: warm=0.000875147s, recompile=0.117337s
- `arf_add` / `service_api_unpadded`: warm=0.000313923s, recompile=0.0289299s
- `arf_add` / `service_api_padded`: warm=0.000843338s, recompile=0.0919328s
- `arf_add` / `service_api_bucketed`: warm=0.000786277s, recompile=0.125727s
- `fmpr_mul` / `service_api_unpadded`: warm=0.000360549s, recompile=0.0386961s
- `fmpr_mul` / `service_api_padded`: warm=0.00221919s, recompile=0.00213052s
- `fmpr_mul` / `service_api_bucketed`: warm=0.00212407s, recompile=0.0400915s
- `arb_fpwrap_double_exp` / `service_api_unpadded`: warm=0.000319564s, recompile=0.0420378s
- `arb_fpwrap_double_exp` / `service_api_padded`: warm=0.000873459s, recompile=0.00113453s
- `arb_fpwrap_double_exp` / `service_api_bucketed`: warm=0.000933956s, recompile=0.064213s
- `acf_mul` / `service_api_unpadded`: warm=0.000439629s, recompile=0.0477663s
- `acf_mul` / `service_api_padded`: warm=0.00100406s, recompile=0.104991s
- `acf_mul` / `service_api_bucketed`: warm=0.00146313s, recompile=0.16014s
- `fmpzi_add` / `service_api_unpadded`: warm=0.000311545s, recompile=0.0637445s
- `fmpzi_add` / `service_api_padded`: warm=0.00105961s, recompile=0.0969224s
- `fmpzi_add` / `service_api_bucketed`: warm=0.000880785s, recompile=0.153837s
