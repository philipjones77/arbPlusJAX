Last updated: 2026-03-18T00:00:00Z

# arb_mat

## Precision Modes

- `point`: first-class midpoint-only matrix path through dedicated `point_wrappers` kernels; intended for fastest JAX execution and independent compilation.
- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: specialized interval-tight path for `det` / `trace`; generic Jacobian/Lipschitz bounds elsewhere.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in `src/arbplusjax/arb_mat.py`. This module exposes direct matrix kernels, precision wrappers, JIT wrappers, and fixed/padded batch helpers.

## Scope

- canonical real interval matrix substrate for Arb-like matrix work
- `n x n` layout contracts plus:
  - `matmul`
  - `matvec`
  - cached `matvec` prepare/apply helpers
  - symmetric-part and SPD structure queries
  - Cholesky / LDL
  - SPD-specialized solve/inverse
  - SPD solve-plan prepare/apply
  - banded `matvec`
  - `solve`
  - `inv`
  - `sqr`
  - `triangular_solve`
  - `lu`
  - `qr`
  - `det`
  - `trace`
  - `norm_fro`
  - `norm_1`
  - `norm_inf`
  - `zero`
  - `identity`
- legacy/specialized `2x2` determinant and trace entry points also remain present

## Intended API Surface

- C reference library: `arb_mat_ref`
  - `arb_mat_2x2_det_ref(a)`
  - `arb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.arb_mat`
  - `arb_mat_as_matrix(a)` for `(..., n, n, 2)`
  - `arb_mat_as_vector(x)` for `(..., n, 2)`
  - `arb_mat_shape(a)`
  - `arb_mat_zero(n)` / `arb_mat_identity(n)`
  - `arb_mat_matmul(a, b)` / `arb_mat_matmul_basic(a, b)` / `arb_mat_matmul_prec(a, b)`
  - `arb_mat_matvec(a, x)` / `arb_mat_matvec_basic(a, x)` / `arb_mat_matvec_prec(a, x)`
  - `arb_mat_matvec_cached_prepare(a)` / `arb_mat_matvec_cached_apply(cache, x)`
  - `arb_mat_symmetric_part(a)` / `arb_mat_is_symmetric(a)` / `arb_mat_is_spd(a)`
  - `arb_mat_cho(a)` / `arb_mat_ldl(a)`
  - `arb_mat_spd_solve(a_or_plan, b)` / `arb_mat_spd_inv(a_or_plan)`
  - `arb_mat_dense_spd_solve_plan_prepare(a)` / `arb_mat_dense_spd_solve_plan_apply(plan, b)`
  - `arb_mat_banded_matvec(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `arb_mat_banded_matvec_basic(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `arb_mat_solve(a, b)` / `arb_mat_solve_basic(a, b)` / `arb_mat_solve_prec(a, b)`
  - `arb_mat_inv(a)` / `arb_mat_inv_basic(a)` / `arb_mat_inv_prec(a)`
  - `arb_mat_sqr(a)` / `arb_mat_sqr_basic(a)` / `arb_mat_sqr_prec(a)`
  - `arb_mat_triangular_solve(a, b, lower=..., unit_diagonal=...)`
  - `arb_mat_triangular_solve_basic(a, b, lower=..., unit_diagonal=...)` / `arb_mat_triangular_solve_prec(...)`
  - `arb_mat_lu(a)` / `arb_mat_lu_basic(a)` / `arb_mat_lu_prec(a)`
  - `arb_mat_qr(a)` / `arb_mat_qr_basic(a)` / `arb_mat_qr_prec(a)`
  - `arb_mat_det(a)` / `arb_mat_det_basic(a)` / `arb_mat_det_rigorous(a)` / `arb_mat_det_prec(a)`
  - `arb_mat_trace(a)` / `arb_mat_trace_basic(a)` / `arb_mat_trace_rigorous(a)` / `arb_mat_trace_prec(a)`
  - `arb_mat_norm_fro(a)` / `arb_mat_norm_fro_basic(a)` / `arb_mat_norm_fro_rigorous(a)`
  - `arb_mat_norm_1(a)` / `arb_mat_norm_1_basic(a)` / `arb_mat_norm_1_rigorous(a)`
  - `arb_mat_norm_inf(a)` / `arb_mat_norm_inf_basic(a)` / `arb_mat_norm_inf_rigorous(a)`
  - `arb_mat_2x2_det(a)`
  - `arb_mat_2x2_trace(a)`
  - JIT aliases such as `arb_mat_matmul_jit`, `arb_mat_solve_jit`, `arb_mat_qr_jit`
  - fixed/padded batch helpers such as `arb_mat_matmul_batch_fixed`, `arb_mat_matmul_batch_padded`, `arb_mat_det_batch_fixed`, `arb_mat_trace_batch_padded`

## Accuracy/Precision Semantics

- matrix entries are interpreted as intervals
- `matmul_basic`, `matvec_basic`, and `banded_matvec_basic` use interval arithmetic directly
- `matvec_cached_prepare` / `matvec_cached_apply` reuse the validated interval matrix layout and currently share the same arithmetic path as `matvec_basic`
- `symmetric_part` is midpoint-based and exact for point intervals
- `is_symmetric` / `is_spd` are midpoint structural diagnostics used to auto-route `solve` and `inv`
- `cho`, `ldl`, `spd_solve`, and `spd_inv` use midpoint Cholesky on the symmetric part, then outward boxing
- `solve` and `solve_basic` both currently use midpoint solve plus outward boxing
- `inv` and `inv_basic` both currently use midpoint inverse plus outward boxing
- `sqr_basic` reuses direct interval `matmul_basic(a, a)`
- `triangular_solve` and `triangular_solve_basic` currently use midpoint triangular solve plus outward boxing
- `lu` / `lu_basic` currently use midpoint LU plus outward boxing of `(P, L, U)`
- `qr` / `qr_basic` currently use midpoint QR plus outward boxing of `(Q, R)`
- `trace_basic` uses direct interval summation on the diagonal
- `det_basic` uses exact interval formulas for `1x1`-`3x3`, then midpoint-plus-outward boxing fallback for larger sizes
- `norm_fro_basic` computes interval sum-of-squares followed by interval `sqrt`
- `norm_1_basic` and `norm_inf_basic` first form interval absolute-value bounds, then row/column sums
- `trace_rigorous` currently aliases the exact interval trace path
- `det_rigorous` currently aliases the tightened `basic` determinant path (`1x1`-`3x3` exact interval formulas, midpoint fallback beyond that)
- `norm_*_rigorous` currently alias the tightened `basic` norm paths
- specialized `2x2 det/trace` entry points keep their existing midpoint/rigorous split

## Differentiability

- Differentiable w.r.t. matrix entries on smooth subdomains.

## Notes

- general matrix contract: `(..., n, n, 2)`
- general vector contract: `(..., n, 2)`
- legacy 2x2 specialized contract: `(..., 2, 2, 2)`
- `zero(n)` and `identity(n)` build point intervals from dense `jax.numpy` constructors
- fixed and padded batch helpers exist for the main hot paths and are intended to keep compile behavior stable under repeated shapes
- generic `solve` / `inv` now auto-select the SPD route when midpoint structure checks pass, so structured dense cases do not require a separate caller-side dispatch

## Current Gaps

- `solve_basic`, `inv_basic`, `triangular_solve_basic`, `lu_basic`, and `qr_basic` are still midpoint-first, not full interval-linear-algebra enclosures
- rigorous coverage is strongest for `trace`, `det`, and the norm helpers; there is not yet a distinct tightened rigorous path for the solve/factorization families
- large-`n` determinant bounds still fall back to midpoint determinant plus outward boxing

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.
