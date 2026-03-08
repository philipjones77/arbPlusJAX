Last updated: 2026-02-25T03:51:38Z

# acb_mat

## Precision Modes

- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: specialized complex-box tight path for `det` / `trace`; generic Jacobian/Lipschitz bounds elsewhere.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="basic|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- canonical complex box matrix substrate for Arb-like matrix work
- `n x n` layout contracts plus:
  - `matmul`
  - `matvec`
  - `solve`
  - `triangular_solve`
  - `lu`
  - `det`
  - `trace`
- legacy/specialized `2x2` determinant and trace entry points also remain present

## Intended API Surface

- C reference library: `acb_mat_ref`
  - `acb_mat_2x2_det_ref(a)`
  - `acb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.acb_mat`
  - `acb_mat_as_matrix(a)` for `(..., n, n, 4)`
  - `acb_mat_as_vector(x)` for `(..., n, 4)`
  - `acb_mat_matmul(a, b)` / `acb_mat_matmul_basic(a, b)`
  - `acb_mat_matvec(a, x)` / `acb_mat_matvec_basic(a, x)`
  - `acb_mat_solve(a, b)` / `acb_mat_solve_basic(a, b)`
  - `acb_mat_triangular_solve(a, b, lower=..., unit_diagonal=...)`
  - `acb_mat_triangular_solve_basic(a, b, lower=..., unit_diagonal=...)`
  - `acb_mat_lu(a)` / `acb_mat_lu_basic(a)`
  - `acb_mat_det(a)` / `acb_mat_det_basic(a)`
  - `acb_mat_trace(a)` / `acb_mat_trace_basic(a)`
  - `acb_mat_2x2_det(a)`
  - `acb_mat_2x2_trace(a)`
  - Precision/jit variants

## Accuracy/Precision Semantics

- matrix entries are interpreted as complex boxes
- `matmul_basic` / `matvec_basic` use box arithmetic directly
- `solve_basic` currently uses midpoint solve plus outward boxing
- `triangular_solve_basic` currently uses midpoint triangular solve plus outward boxing
- `lu_basic` currently uses midpoint LU plus outward boxing of `(P, L, U)`
- `trace_basic` uses direct complex-box summation on the diagonal
- `det_basic` uses exact complex-box formulas for `1x1`-`3x3`, then midpoint-plus-outward boxing fallback for larger sizes
- `trace_rigorous` currently aliases the exact complex-box trace path
- `det_rigorous` currently aliases the tightened `basic` determinant path (`1x1`-`3x3` exact complex-box formulas, midpoint fallback beyond that)
- specialized `2x2 det/trace` entry points keep their existing midpoint/rigorous split

## Differentiability

- Differentiable w.r.t. matrix entries for smooth subdomains.

## Notes

- general matrix contract: `(..., n, n, 4)`
- general vector contract: `(..., n, 4)`
- legacy 2x2 specialized contract: `(..., 2, 2, 4)`

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.

## Implementation Notes

- Uses complex-box arithmetic on entries.
