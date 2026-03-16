Last updated: 2026-03-13T00:00:00Z

# acb_mat

## Precision Modes

- `point`: first-class midpoint-only complex-matrix path for independent compilation and the fastest dense box substrate.
- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: specialized complex-box tight path for `det` / `trace`; generic Jacobian/Lipschitz bounds elsewhere.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in `src/arbplusjax/acb_mat.py`. This module exposes direct box-valued kernels, precision wrappers, JIT wrappers, and fixed/padded batch helpers.

## Scope

- canonical complex box matrix substrate for Arb-like matrix work
- `n x n` layout contracts plus:
  - `matmul`
  - `matvec`
  - cached `matvec` prepare/apply helpers
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

- C reference library: `acb_mat_ref`
  - `acb_mat_2x2_det_ref(a)`
  - `acb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.acb_mat`
  - `acb_mat_as_matrix(a)` for `(..., n, n, 4)`
  - `acb_mat_as_vector(x)` for `(..., n, 4)`
  - `acb_mat_shape(a)`
  - `acb_mat_zero(n)` / `acb_mat_identity(n)`
  - `acb_mat_matmul(a, b)` / `acb_mat_matmul_basic(a, b)` / `acb_mat_matmul_prec(a, b)`
  - `acb_mat_matvec(a, x)` / `acb_mat_matvec_basic(a, x)` / `acb_mat_matvec_prec(a, x)`
  - `acb_mat_matvec_cached_prepare(a)` / `acb_mat_matvec_cached_apply(cache, x)`
  - `acb_mat_banded_matvec(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `acb_mat_banded_matvec_basic(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `acb_mat_solve(a, b)` / `acb_mat_solve_basic(a, b)` / `acb_mat_solve_prec(a, b)`
  - `acb_mat_inv(a)` / `acb_mat_inv_basic(a)` / `acb_mat_inv_prec(a)`
  - `acb_mat_sqr(a)` / `acb_mat_sqr_basic(a)` / `acb_mat_sqr_prec(a)`
  - `acb_mat_triangular_solve(a, b, lower=..., unit_diagonal=...)`
  - `acb_mat_triangular_solve_basic(a, b, lower=..., unit_diagonal=...)` / `acb_mat_triangular_solve_prec(...)`
  - `acb_mat_lu(a)` / `acb_mat_lu_basic(a)` / `acb_mat_lu_prec(a)`
  - `acb_mat_qr(a)` / `acb_mat_qr_basic(a)` / `acb_mat_qr_prec(a)`
  - `acb_mat_det(a)` / `acb_mat_det_basic(a)` / `acb_mat_det_rigorous(a)` / `acb_mat_det_prec(a)`
  - `acb_mat_trace(a)` / `acb_mat_trace_basic(a)` / `acb_mat_trace_rigorous(a)` / `acb_mat_trace_prec(a)`
  - `acb_mat_norm_fro(a)` / `acb_mat_norm_fro_basic(a)` / `acb_mat_norm_fro_rigorous(a)`
  - `acb_mat_norm_1(a)` / `acb_mat_norm_1_basic(a)` / `acb_mat_norm_1_rigorous(a)`
  - `acb_mat_norm_inf(a)` / `acb_mat_norm_inf_basic(a)` / `acb_mat_norm_inf_rigorous(a)`
  - `acb_mat_2x2_det(a)`
  - `acb_mat_2x2_trace(a)`
  - JIT aliases such as `acb_mat_matmul_jit`, `acb_mat_solve_jit`, `acb_mat_qr_jit`
  - fixed/padded batch helpers such as `acb_mat_matmul_batch_fixed`, `acb_mat_matmul_batch_padded`, `acb_mat_det_batch_fixed`, `acb_mat_trace_batch_padded`

## Accuracy/Precision Semantics

- matrix entries are interpreted as complex boxes
- `matmul_basic`, `matvec_basic`, and `banded_matvec_basic` use box arithmetic directly
- `matvec_cached_prepare` / `matvec_cached_apply` reuse the validated box layout and currently share the same arithmetic path as `matvec_basic`
- `solve` and `solve_basic` both currently use midpoint solve plus outward boxing
- `inv` and `inv_basic` both currently use midpoint inverse plus outward boxing
- `sqr_basic` reuses direct box `matmul_basic(a, a)`
- `triangular_solve` and `triangular_solve_basic` currently use midpoint triangular solve plus outward boxing
- `lu` / `lu_basic` currently use midpoint LU plus outward boxing of `(P, L, U)`
- `qr` / `qr_basic` currently use midpoint QR plus outward boxing of `(Q, R)`
- `trace_basic` uses direct complex-box summation on the diagonal
- `det_basic` uses exact complex-box formulas for `1x1`-`3x3`, then midpoint-plus-outward boxing fallback for larger sizes
- `norm_fro_basic` computes a box-valued sum of `a * conj(a)` followed by `acb_sqrt`
- `norm_1_basic` and `norm_inf_basic` derive interval absolute-value bounds first, then row/column sums
- `trace_rigorous` currently aliases the exact complex-box trace path
- `det_rigorous` currently aliases the tightened `basic` determinant path (`1x1`-`3x3` exact complex-box formulas, midpoint fallback beyond that)
- `norm_*_rigorous` currently alias the tightened `basic` norm paths
- specialized `2x2 det/trace` entry points keep their existing midpoint/rigorous split

## Differentiability

- Differentiable w.r.t. matrix entries for smooth subdomains.

## Notes

- general matrix contract: `(..., n, n, 4)`
- general vector contract: `(..., n, 4)`
- legacy 2x2 specialized contract: `(..., 2, 2, 4)`
- `zero(n)` and `identity(n)` construct point boxes from dense complex `jax.numpy` arrays
- fixed and padded batch helpers exist for the main hot paths and are intended to reduce shape-driven recompilation

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.

## Current Gaps

- `solve_basic`, `inv_basic`, `triangular_solve_basic`, `lu_basic`, and `qr_basic` are still midpoint-first, not full complex-box enclosure algorithms
- rigorous coverage is strongest for `trace`, `det`, and the norm helpers; solve/factorization families do not yet have distinct tightened rigorous implementations
- large-`n` determinant bounds still fall back to midpoint determinant plus outward boxing
