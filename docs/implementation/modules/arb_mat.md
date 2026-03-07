Last updated: 2026-02-25T03:51:38Z

# arb_mat

## Precision Modes

- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="basic|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- canonical real interval matrix substrate for Arb-like matrix work
- `n x n` layout contracts plus:
  - `matmul`
  - `matvec`
  - `solve`
- legacy/specialized `2x2` determinant and trace entry points remain present

## Intended API Surface

- C reference library: `arb_mat_ref`
  - `arb_mat_2x2_det_ref(a)`
  - `arb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.arb_mat`
  - `arb_mat_as_matrix(a)` for `(..., n, n, 2)`
  - `arb_mat_as_vector(x)` for `(..., n, 2)`
  - `arb_mat_matmul(a, b)` / `arb_mat_matmul_basic(a, b)`
  - `arb_mat_matvec(a, x)` / `arb_mat_matvec_basic(a, x)`
  - `arb_mat_solve(a, b)` / `arb_mat_solve_basic(a, b)`
  - `arb_mat_2x2_det(a)`
  - `arb_mat_2x2_trace(a)`
  - Precision/jit variants

## Accuracy/Precision Semantics

- matrix entries are interpreted as intervals
- `matmul_basic` / `matvec_basic` use interval arithmetic directly
- `solve_basic` currently uses midpoint solve plus outward boxing
- `2x2 det/trace` keep their existing midpoint/rigorous split

## Differentiability

- Differentiable w.r.t. matrix entries on smooth subdomains.

## Notes

- general matrix contract: `(..., n, n, 2)`
- general vector contract: `(..., n, 2)`
- legacy 2x2 specialized contract: `(..., 2, 2, 2)`

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.
