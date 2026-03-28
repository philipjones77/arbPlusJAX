Last updated: 2026-03-26T00:00:00Z

# Matrix-Free First Use Inventory

This report records the observed `arbplusjax.*` module set for representative matrix-free first-use boundaries.

Budgets:
- operator creation budget: `<= 2`
- operator apply budget: `<= 2`
- Krylov solve budget: `<= 5`
- implicit-adjoint solve budget: `<= 5`
- estimator first use budget: `<= 3`
- contour first use budget: `<= 3`
- real SLQ wrapper first use budget: `<= 18`
- complex SLQ wrapper first use budget: `<= 18`
- real Hutch++ wrapper first use budget: `<= 19`
- complex Hutch++ wrapper first use budget: `<= 19`

## Operator Creation (`dense_operator_plan(...)`)

- observed module count: `2`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_krylov` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.iterative_solvers` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision"
]
```

## Operator Apply (`operator_plan_apply(...)`)

- observed module count: `2`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_krylov` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.iterative_solvers` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision"
]
```

## Krylov Solve (`krylov_solve_midpoint(...)`)

- observed module count: `5`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_krylov` loaded: `True`
- `arbplusjax.iterative_solvers` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.matrix_free_core",
  "arbplusjax.matrix_free_krylov",
  "arbplusjax.precision"
]
```

## Implicit-Adjoint Solve (`implicit_krylov_solve_midpoint(...)`)

- observed module count: `5`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_adjoint` loaded: `True`
- `arbplusjax.iterative_solvers` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_krylov` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.matrix_free_adjoint",
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision"
]
```

## Estimator First Use (`make_probe_estimate_statistics(...)`)

- observed module count: `3`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `True`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_krylov` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.iterative_solvers` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.matrix_free_core",
  "arbplusjax.matrix_free_estimators",
  "arbplusjax.precision"
]
```

## Contour First Use (`contour_integral_action_point(...)`)

- observed module count: `3`
- `arbplusjax.matrix_free_core` loaded: `True`
- `arbplusjax.matrix_free_contour` loaded: `True`
- `arbplusjax.matrix_free_estimators` loaded: `False`
- `arbplusjax.matrix_free_krylov` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.iterative_solvers` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.matrix_free_contour",
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision"
]
```

## Real SLQ Wrapper First Use (`jrb_mat_logdet_estimate_point(...)`)

- observed module count: `18`
- `arbplusjax.jrb_mat` loaded: `True`
- `arbplusjax.jrb_mat_slq_wrappers` loaded: `True`
- `arbplusjax.jrb_mat_contour_wrappers` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.acb_core",
  "arbplusjax.arb_core",
  "arbplusjax.checks",
  "arbplusjax.double_interval",
  "arbplusjax.elementary",
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.jrb_mat",
  "arbplusjax.jrb_mat_slq_wrappers",
  "arbplusjax.kernel_helpers",
  "arbplusjax.lazy_jit",
  "arbplusjax.mat_common",
  "arbplusjax.matrix_free_basic",
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision",
  "arbplusjax.sparse_common",
  "arbplusjax.srb_block_mat",
  "arbplusjax.srb_vblock_mat"
]
```

## Complex SLQ Wrapper First Use (`jcb_mat_logdet_estimate_point(...)`)

- observed module count: `18`
- `arbplusjax.jcb_mat` loaded: `True`
- `arbplusjax.jcb_mat_slq_wrappers` loaded: `True`
- `arbplusjax.jcb_mat_contour_wrappers` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.acb_core",
  "arbplusjax.arb_core",
  "arbplusjax.checks",
  "arbplusjax.double_interval",
  "arbplusjax.elementary",
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.jcb_mat",
  "arbplusjax.jcb_mat_slq_wrappers",
  "arbplusjax.kernel_helpers",
  "arbplusjax.lazy_jit",
  "arbplusjax.mat_common",
  "arbplusjax.matrix_free_basic",
  "arbplusjax.matrix_free_core",
  "arbplusjax.precision",
  "arbplusjax.scb_block_mat",
  "arbplusjax.scb_vblock_mat",
  "arbplusjax.sparse_common"
]
```

## Real Hutch++ Wrapper First Use (`jrb_mat_hutchpp_trace_point(...)`)

- observed module count: `19`
- `arbplusjax.jrb_mat` loaded: `True`
- `arbplusjax.jrb_mat_hutchpp_wrappers` loaded: `True`
- `arbplusjax.jrb_mat_contour_wrappers` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.acb_core",
  "arbplusjax.arb_core",
  "arbplusjax.checks",
  "arbplusjax.double_interval",
  "arbplusjax.elementary",
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.jrb_mat",
  "arbplusjax.jrb_mat_hutchpp_wrappers",
  "arbplusjax.kernel_helpers",
  "arbplusjax.lazy_jit",
  "arbplusjax.mat_common",
  "arbplusjax.matrix_free_basic",
  "arbplusjax.matrix_free_core",
  "arbplusjax.matrix_free_estimators",
  "arbplusjax.precision",
  "arbplusjax.sparse_common",
  "arbplusjax.srb_block_mat",
  "arbplusjax.srb_vblock_mat"
]
```

## Complex Hutch++ Wrapper First Use (`jcb_mat_hutchpp_trace_point(...)`)

- observed module count: `19`
- `arbplusjax.jcb_mat` loaded: `True`
- `arbplusjax.jcb_mat_hutchpp_wrappers` loaded: `True`
- `arbplusjax.jcb_mat_contour_wrappers` loaded: `False`
- `arbplusjax.matrix_free_contour` loaded: `False`
- `arbplusjax.matrix_free_adjoint` loaded: `False`
- `arbplusjax.matfree_adjoints` loaded: `False`

```json
[
  "arbplusjax.acb_core",
  "arbplusjax.arb_core",
  "arbplusjax.checks",
  "arbplusjax.double_interval",
  "arbplusjax.elementary",
  "arbplusjax.iterative_solvers",
  "arbplusjax.jax_precision",
  "arbplusjax.jcb_mat",
  "arbplusjax.jcb_mat_hutchpp_wrappers",
  "arbplusjax.kernel_helpers",
  "arbplusjax.lazy_jit",
  "arbplusjax.mat_common",
  "arbplusjax.matrix_free_basic",
  "arbplusjax.matrix_free_core",
  "arbplusjax.matrix_free_estimators",
  "arbplusjax.precision",
  "arbplusjax.scb_block_mat",
  "arbplusjax.scb_vblock_mat",
  "arbplusjax.sparse_common"
]
```

## Notes

- plain operator creation and primitive apply should stay on `matrix_free_core` only
- Krylov solve should load the Krylov runtime layer without loading the implicit-adjoint runtime layer
- implicit-adjoint solve should load the implicit-adjoint runtime layer without loading `matfree_adjoints` helper machinery
- estimator helpers should load `matrix_free_estimators` and contour helpers should load `matrix_free_contour` without widening the operator-only path
- real and complex SLQ wrapper first use should load only the selected wrapper module, not the contour or implicit-adjoint wrapper families
- real and complex Hutch++ wrapper first use should load only the selected wrapper module, not the contour or implicit-adjoint wrapper families
- `matfree_adjoints` remains lazy and should load only when the explicit adjoint helper surface is selected
