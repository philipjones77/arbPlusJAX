Last updated: 2026-03-26T00:00:00Z

# Entry Script Startup Inventory

This report measures benchmark/example entry-script startup after the lazy-entry refactor and records where top-level `arbplusjax` imports still remain.

Interpretation:
- `top-level arbplusjax imports` means the script still pulls repo family modules during module import before first real use
- `arbplusjax module count` records how much of the repo was loaded just to reach `--help` or import-only startup
- high startup with zero top-level `arbplusjax` imports usually means the remaining delay is mostly JAX/Python/runtime bootstrap cost, not repo family import debt

| path | mode | mean startup s | min s | max s | arbplusjax modules | top-level arbplusjax imports | top-level jax imports |
|---|---:|---:|---:|---:|---:|---|---|
| [benchmarks/benchmark_api_surface.py](/benchmarks/benchmark_api_surface.py) | `help` | `0.457` | `0.421` | `0.516` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_matrix_service_api.py](/benchmarks/benchmark_matrix_service_api.py) | `help` | `0.480` | `0.434` | `0.531` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_special_function_service_api.py](/benchmarks/benchmark_special_function_service_api.py) | `help` | `0.412` | `0.407` | `0.421` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_matrix_free_krylov.py](/benchmarks/benchmark_matrix_free_krylov.py) | `help` | `0.422` | `0.404` | `0.451` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_core_scalar_service_api.py](/benchmarks/benchmark_core_scalar_service_api.py) | `help` | `0.404` | `0.401` | `0.410` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_dense_matrix_surface.py](/benchmarks/benchmark_dense_matrix_surface.py) | `help` | `0.397` | `0.391` | `0.404` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_sparse_matrix_surface.py](/benchmarks/benchmark_sparse_matrix_surface.py) | `help` | `0.422` | `0.392` | `0.440` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_block_sparse_matrix_surface.py](/benchmarks/benchmark_block_sparse_matrix_surface.py) | `help` | `0.435` | `0.400` | `0.473` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_vblock_sparse_matrix_surface.py](/benchmarks/benchmark_vblock_sparse_matrix_surface.py) | `help` | `0.412` | `0.384` | `0.433` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_matrix_stack_diagnostics.py](/benchmarks/benchmark_matrix_stack_diagnostics.py) | `help` | `0.414` | `0.375` | `0.464` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_arb_mat.py](/benchmarks/benchmark_arb_mat.py) | `help` | `0.402` | `0.379` | `0.434` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_acb_calc.py](/benchmarks/benchmark_acb_calc.py) | `help` | `0.428` | `0.390` | `0.473` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_barnes_double_gamma.py](/benchmarks/benchmark_barnes_double_gamma.py) | `help` | `0.423` | `0.370` | `0.473` | `0` | `-` | `jax, jax.numpy` |
| [benchmarks/benchmark_hypgeom.py](/benchmarks/benchmark_hypgeom.py) | `help` | `error` | `-` | `-` | `-` | `-` | `jax.numpy` |
| [benchmarks/compare_arb_mat.py](/benchmarks/compare_arb_mat.py) | `help` | `0.401` | `0.364` | `0.434` | `0` | `-` | `jax.numpy` |
| [benchmarks/compare_acb_calc.py](/benchmarks/compare_acb_calc.py) | `help` | `0.418` | `0.409` | `0.427` | `0` | `-` | `jax.numpy` |
| [examples/example_latent_gaussian_laplace.py](/examples/example_latent_gaussian_laplace.py) | `import` | `0.428` | `0.409` | `0.452` | `0` | `-` | `jax, jax.numpy` |

## Remaining Top-Level `arbplusjax` Import Debt

- scripts still carrying top-level `arbplusjax` imports across `benchmarks/` and `examples/`: `58`
- [benchmarks/benchmark_acb_dirichlet.py](/benchmarks/benchmark_acb_dirichlet.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_acb_mat.py](/benchmarks/benchmark_acb_mat.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_acb_modular.py](/benchmarks/benchmark_acb_modular.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_acb_poly.py](/benchmarks/benchmark_acb_poly.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_acf.py](/benchmarks/benchmark_acf.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_arb_calc.py](/benchmarks/benchmark_arb_calc.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_arb_fmpz_poly.py](/benchmarks/benchmark_arb_fmpz_poly.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_arb_fpwrap.py](/benchmarks/benchmark_arb_fpwrap.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_arb_poly.py](/benchmarks/benchmark_arb_poly.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_arf.py](/benchmarks/benchmark_arf.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_bernoulli.py](/benchmarks/benchmark_bernoulli.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_bool_mat.py](/benchmarks/benchmark_bool_mat.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_core_scalar_batch_padding.py](/benchmarks/benchmark_core_scalar_batch_padding.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_dirichlet.py](/benchmarks/benchmark_dirichlet.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_dlog.py](/benchmarks/benchmark_dlog.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_fft_nufft.py](/benchmarks/benchmark_fft_nufft.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_fmpr.py](/benchmarks/benchmark_fmpr.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_fmpz_extras.py](/benchmarks/benchmark_fmpz_extras.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_fmpzi.py](/benchmarks/benchmark_fmpzi.py) : `1` top-level `arbplusjax` import statements
- [benchmarks/benchmark_gamma_compare.py](/benchmarks/benchmark_gamma_compare.py) : `1` top-level `arbplusjax` import statements

## Assessment

- import-boundary debt still exists in `58` benchmark/example scripts that keep top-level `arbplusjax` imports
- in this measured entry set, `0` scripts still have direct top-level `arbplusjax` imports and should be treated as remaining repo import debt
- in this measured entry set, `0` scripts have zero top-level `arbplusjax` imports but still take at least `0.6s` to start; those are now mostly dominated by JAX import/backend/runtime bootstrap cost
- `1` measured entry scripts currently fail before timing completes because of missing optional dependencies or other local runtime issues
- `--help` paths that still import `jax` at module top level will continue to pay significant cold-start cost even after repo-family lazy-loading is fixed

## Measurement Failures

- [benchmarks/benchmark_hypgeom.py](/benchmarks/benchmark_hypgeom.py) : `ModuleNotFoundError: No module named 'psutil'`
