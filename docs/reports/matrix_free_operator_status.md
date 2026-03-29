Last updated: 2026-03-29T00:00:00Z

# Matrix-Free Operator Status

## Scope

- [matrix_free_completion_plan.md](/docs/status/matrix_free_completion_plan.md)
- [matrix_free_operator_methods.md](/docs/theory/matrix_free_operator_methods.md)
- [matrix_free.md](/docs/practical/matrix_free.md)
- [benchmark_matrix_free_krylov.py](/benchmarks/benchmark_matrix_free_krylov.py)
- [example_matrix_free_operator_surface.ipynb](/examples/example_matrix_free_operator_surface.ipynb)

## Status Table

| family | point | basic | adaptive | rigorous | fast JAX | practical JAX | diagnostics | AD | notes |
|---|---|---|---|---|---|---|---|---|---|
| `jrb_mat` operator-first real matrix-free | yes | yes | partial | partial | dedicated `*_point_jit` plan-safe kernels for apply, transpose-apply, function action, SLQ logdet/det, solve, inverse, eigensolvers, and multi-shift reuse | prepared plan reuse, compile/execute split metrics, `choose_matrix_free_plan_policy(...)`, and `prewarm_matrix_free_kernels(...)` now cover apply, cached transpose apply, preconditioner reuse, and diagnostics separation | `*_with_diagnostics_point/basic` and eigensolver diagnostics are public and remain off the repeated hot path | argument- and parameter-direction proof coverage exists on solve/logdet-facing operator-plan surfaces | real structured aliases remain the preferred production path for symmetric / SPD workloads |
| `jcb_mat` operator-first complex matrix-free | yes | yes | partial | partial | dedicated `*_point_jit` plan-safe kernels for apply, transpose-apply, adjoint apply, function action, SLQ logdet/det, solve, inverse, eigensolvers, and multi-shift reuse | prepared plan reuse, compile/execute split metrics, `choose_matrix_free_plan_policy(...)`, and `prewarm_matrix_free_kernels(...)` now cover apply, transpose / adjoint apply, preconditioner reuse, and diagnostics separation | `*_with_diagnostics_point/basic` and eigensolver diagnostics are public and remain off the repeated hot path | argument- and parameter-direction proof coverage exists on solve/logdet-facing operator-plan surfaces | Hermitian / HPD aliases remain the preferred production path for stable Krylov and solve selection |

## Current Readout

- operator-plan apply, transpose-apply, and adjoint-apply surfaces are exposed and covered in the public Jones wrappers
- dedicated plan-safe `jit` entrypoints exist for repeated solve, inverse, SLQ logdet/det, restarted actions, and multi-shift solves
- point and basic are the completed public matrix-free modes in the current repo state; adaptive and rigorous remain partial rather than being claimed as complete
- diagnostics exist across solve, estimator, and eigensolver families and are intentionally separated from the hot repeated-call kernels
- the canonical notebook and benchmark now explicitly show cached transpose / adjoint plan reuse alongside solve/logdet and compile/execute splits
- the public API now exposes backend-policy and startup-prewarm helpers for repeated matrix-free workloads instead of leaving GPU warmup as an undocumented manual step

## Backend Interpretation

- CPU remains the default recommendation for small dense repeated matrix-free workloads in the retained benchmark
- GPU is now retained-validated in the current WSL `jax` environment for matrix-free benchmarking and notebook execution
- the main GPU wins currently show up on sparse complex operator-plan workloads rather than tiny dense plan workloads
- practical optimization therefore means:
  - keep operator-plan payloads stable
  - reuse prepared plans and preconditioners
  - choose one backend policy for a repeated workload
  - prewarm the exact apply / solve / logdet / multi-shift kernels you intend to reuse
  - keep Krylov step counts and solver flags fixed across loops
  - keep diagnostics off the repeated hot path

## Remaining Hard Work

- broader adaptive / rigorous operator semantics
- fuller flexible-preconditioner policy beyond the current landed shell / Jacobi / transpose-aware path
- further lowering GPU compile cost on dense real Krylov-heavy paths beyond prewarming and shape stability
