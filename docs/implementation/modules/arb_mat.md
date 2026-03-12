Last updated: 2026-02-25T03:51:38Z

# arb_mat

## Precision Modes

- `point`: first-class midpoint-only matrix path through dedicated `point_wrappers` kernels; intended for fastest JAX execution and independent compilation.
- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: specialized interval-tight path for `det` / `trace`; generic Jacobian/Lipschitz bounds elsewhere.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="basic|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- canonical real interval matrix substrate for Arb-like matrix work
- `n x n` layout contracts plus:
  - `matmul`
  - `matvec`
  - `solve`
  - `inv`
  - `triangular_solve`
  - `lu`
  - `qr`
  - `det`
  - `trace`
- legacy/specialized `2x2` determinant and trace entry points also remain present

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
  - `arb_mat_inv(a)` / `arb_mat_inv_basic(a)`
  - `arb_mat_triangular_solve(a, b, lower=..., unit_diagonal=...)`
  - `arb_mat_triangular_solve_basic(a, b, lower=..., unit_diagonal=...)`
  - `arb_mat_lu(a)` / `arb_mat_lu_basic(a)`
  - `arb_mat_qr(a)` / `arb_mat_qr_basic(a)`
  - `arb_mat_det(a)` / `arb_mat_det_basic(a)`
  - `arb_mat_trace(a)` / `arb_mat_trace_basic(a)`
  - `arb_mat_2x2_det(a)`
  - `arb_mat_2x2_trace(a)`
  - Precision/jit variants

## Accuracy/Precision Semantics

- matrix entries are interpreted as intervals
- `matmul_basic` / `matvec_basic` use interval arithmetic directly
- `solve_basic` currently uses midpoint solve plus outward boxing
- `inv_basic` currently uses midpoint inverse plus outward boxing
- `triangular_solve_basic` currently uses midpoint triangular solve plus outward boxing
- `lu_basic` currently uses midpoint LU plus outward boxing of `(P, L, U)`
- `qr_basic` currently uses midpoint QR plus outward boxing of `(Q, R)`
- `trace_basic` uses direct interval summation on the diagonal
- `det_basic` uses exact interval formulas for `1x1`-`3x3`, then midpoint-plus-outward boxing fallback for larger sizes
- `trace_rigorous` currently aliases the exact interval trace path
- `det_rigorous` currently aliases the tightened `basic` determinant path (`1x1`-`3x3` exact interval formulas, midpoint fallback beyond that)
- specialized `2x2 det/trace` entry points keep their existing midpoint/rigorous split

## Differentiability

- Differentiable w.r.t. matrix entries on smooth subdomains.

## Notes

- general matrix contract: `(..., n, n, 2)`
- general vector contract: `(..., n, 2)`
- legacy 2x2 specialized contract: `(..., 2, 2, 2)`

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.
