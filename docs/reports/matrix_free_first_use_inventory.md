Last updated: 2026-03-26T00:00:00Z

# Matrix-Free First Use Inventory

This report records the observed `arbplusjax.*` module set for representative matrix-free first-use boundaries.

Budgets:
- operator creation budget: `<= 2`
- operator apply budget: `<= 2`
- Krylov solve budget: `<= 5`
- implicit-adjoint solve budget: `<= 5`

## Operator Creation (`dense_operator_plan(...)`)

- observed module count: `2`
- `arbplusjax.matrix_free_core` loaded: `True`
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

## Notes

- plain operator creation and primitive apply should stay on `matrix_free_core` only
- Krylov solve should load the Krylov runtime layer without loading the implicit-adjoint runtime layer
- implicit-adjoint solve should load the implicit-adjoint runtime layer without loading `matfree_adjoints` helper machinery
- `matfree_adjoints` remains lazy and should load only when the explicit adjoint helper surface is selected
