Last updated: 2026-03-27T00:00:00Z

# Point Fast JAX Verification

This report verifies the repo-wide `fast JAX` contract for public `point` mode only.

Policy references:
- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)
- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)
- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)

Verification rule:
- public point surface exists
- `api.eval_point(..., jit=True)` exists
- `api.bind_point_batch_jit(...)` exists
- representative category-owned proof test exists
- benchmark and canonical notebook evidence exists for the category

Total public point functions: `958`
Direct family-owned point batch fastpaths: `958`

## Family Counts

- `barnes`: `2` public point functions
- `bessel`: `16` public point functions
- `core`: `155` public point functions
- `gamma`: `47` public point functions
- `hypergeometric`: `312` public point functions
- `integration`: `4` public point functions
- `matrix`: `422` public point functions

## Category Verification

| category | covered families | point_count | compiled_single | compiled_batch | direct_fastpath_family_presence | evidence_status | proof tests | benchmarks | notebooks |
|---|---|---|---|---|---|---|---|---|---|
| `1. core numeric scalars` | `core` | `155` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_core_scalar_api_contracts.py](/tests/test_core_scalar_api_contracts.py) | [benchmark_core_scalar_service_api.py](/benchmarks/benchmark_core_scalar_service_api.py), [benchmark_api_surface.py](/benchmarks/benchmark_api_surface.py) | [example_core_scalar_surface.ipynb](/examples/example_core_scalar_surface.ipynb), [example_api_surface.ipynb](/examples/example_api_surface.ipynb) |
| `2. interval / box / precision modes` | `core` | `155` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_dense_plan_modes.py](/tests/test_dense_plan_modes.py), [test_arb_core_top10_modes.py](/tests/test_arb_core_top10_modes.py) | [benchmark_api_surface.py](/benchmarks/benchmark_api_surface.py) | [example_core_scalar_surface.ipynb](/examples/example_core_scalar_surface.ipynb), [example_api_surface.ipynb](/examples/example_api_surface.ipynb) |
| `3. dense matrix functionality` | `matrix` | `422` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_arb_mat_chassis.py](/tests/test_arb_mat_chassis.py), [test_acb_mat_chassis.py](/tests/test_acb_mat_chassis.py) | [benchmark_dense_matrix_surface.py](/benchmarks/benchmark_dense_matrix_surface.py) | [example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb) |
| `4. sparse / block-sparse / vblock functionality` | `matrix` | `422` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_sparse_format_modes.py](/tests/test_sparse_format_modes.py) | [benchmark_sparse_matrix_surface.py](/benchmarks/benchmark_sparse_matrix_surface.py), [benchmark_block_sparse_matrix_surface.py](/benchmarks/benchmark_block_sparse_matrix_surface.py), [benchmark_vblock_sparse_matrix_surface.py](/benchmarks/benchmark_vblock_sparse_matrix_surface.py) | [example_sparse_matrix_surface.ipynb](/examples/example_sparse_matrix_surface.ipynb) |
| `5. matrix-free / operator functionality` | `matrix` | `422` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_matrix_free_logdet_solve_jit.py](/tests/test_matrix_free_logdet_solve_jit.py), [test_matrix_free_basic.py](/tests/test_matrix_free_basic.py) | [benchmark_matrix_free_krylov.py](/benchmarks/benchmark_matrix_free_krylov.py) | [example_matrix_free_operator_surface.ipynb](/examples/example_matrix_free_operator_surface.ipynb) |
| `6. special functions` | `barnes`, `bessel`, `gamma`, `hypergeometric`, `integration` | `381` | `yes` | `yes` | `yes` | `complete` | [test_point_fast_jax_categories.py](/tests/test_point_fast_jax_categories.py), [test_special_function_hardening.py](/tests/test_special_function_hardening.py), [test_special_function_ad_directions.py](/tests/test_special_function_ad_directions.py) | [benchmark_special_function_service_api.py](/benchmarks/benchmark_special_function_service_api.py), [special_function_ad_benchmark.py](/benchmarks/special_function_ad_benchmark.py), [special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py) | [example_gamma_family_surface.ipynb](/examples/example_gamma_family_surface.ipynb), [example_barnes_double_gamma_surface.ipynb](/examples/example_barnes_double_gamma_surface.ipynb), [example_hypgeom_family_surface.ipynb](/examples/example_hypgeom_family_surface.ipynb) |

## Current Verification Note

- This report is intentionally point-only. It does not verify `basic`, `adaptive`, or `rigorous` mode semantics.
- The verification proof is category-owned rather than one giant smoke test, so each major runtime family keeps its own fast-JAX evidence near the implementation.
- Deeper per-function numerical hardening still belongs in the owning family tests and benchmarks; this report verifies the public point fast-JAX contract, not full mathematical completeness.
