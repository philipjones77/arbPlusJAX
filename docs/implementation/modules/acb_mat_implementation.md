Last updated: 2026-03-18T00:00:00Z

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
  - `add`
  - `sub`
  - `neg`
  - entrywise multiply
  - conjugation
  - `charpoly`
  - `pow_ui`
  - `exp`
  - `matmul`
  - `matvec`
  - cached `matvec` prepare/apply helpers
  - Hermitian-part and HPD structure queries
  - diagonal / triangular / zero / finite / exact / real-valued predicates
  - Hermitian Cholesky / LDL
  - Hermitian eigenspectrum / eigendecomposition
  - HPD-specialized solve/inverse
  - HPD solve-plan prepare/apply
  - banded `matvec`
  - `solve`
  - `solve_tril`
  - `solve_triu`
  - `solve_lu`
  - `solve_transpose`
  - `solve_add`
  - `solve_transpose_add`
  - `mat_solve`
  - `mat_solve_transpose`
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
  - named constructors: companion / Hilbert / Pascal / Stirling
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
  - `acb_mat_add(a, b)` / `acb_mat_add_prec(a, b)`
  - `acb_mat_sub(a, b)` / `acb_mat_sub_prec(a, b)`
  - `acb_mat_neg(a)` / `acb_mat_neg_prec(a)`
  - `acb_mat_mul_entrywise(a, b)` / `acb_mat_mul_entrywise_prec(a, b)`
  - `acb_mat_conjugate(a)` / `acb_mat_conjugate_prec(a)`
  - `acb_mat_charpoly(a)` / `acb_mat_charpoly_prec(a)`
  - `acb_mat_pow_ui(a, n)` / `acb_mat_pow_ui_prec(a, n)`
  - `acb_mat_exp(a)` / `acb_mat_exp_prec(a)`
  - `acb_mat_matmul(a, b)` / `acb_mat_matmul_basic(a, b)` / `acb_mat_matmul_prec(a, b)`
  - `acb_mat_matvec(a, x)` / `acb_mat_matvec_basic(a, x)` / `acb_mat_matvec_prec(a, x)`
  - `acb_mat_matvec_cached_prepare(a)` / `acb_mat_matvec_cached_apply(cache, x)`
  - `acb_mat_hermitian_part(a)` / `acb_mat_is_hermitian(a)` / `acb_mat_is_hpd(a)`
  - `acb_mat_is_diag(a)` / `acb_mat_is_tril(a)` / `acb_mat_is_triu(a)`
  - `acb_mat_is_zero(a)` / `acb_mat_is_finite(a)` / `acb_mat_is_exact(a)` / `acb_mat_is_real(a)`
  - `acb_mat_cho(a)` / `acb_mat_ldl(a)`
  - `acb_mat_eigvalsh(a)` / `acb_mat_eigh(a)`
  - `acb_mat_hpd_solve(a_or_plan, b)` / `acb_mat_hpd_inv(a_or_plan)`
  - `acb_mat_dense_hpd_solve_plan_prepare(a)` / `acb_mat_dense_hpd_solve_plan_apply(plan, b)`
  - `acb_mat_banded_matvec(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `acb_mat_banded_matvec_basic(a, x, lower_bandwidth=..., upper_bandwidth=...)`
  - `acb_mat_solve(a, b)` / `acb_mat_solve_basic(a, b)` / `acb_mat_solve_prec(a, b)`
  - `acb_mat_solve_tril(a, b, unit_diagonal=...)` / `acb_mat_solve_triu(a, b, unit_diagonal=...)`
  - `acb_mat_solve_lu(a_or_plan, b)` / `acb_mat_solve_lu_precomp(plan, b)`
  - `acb_mat_solve_transpose(a_or_plan, b)`
  - `acb_mat_solve_add(a_or_plan, b, y)` / `acb_mat_solve_transpose_add(a_or_plan, b, y)`
  - `acb_mat_mat_solve(a_or_plan, b)` / `acb_mat_mat_solve_transpose(a_or_plan, b)`
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
  - `acb_mat_companion(coeffs)` / `acb_mat_hilbert(n)` / `acb_mat_pascal(n)` / `acb_mat_stirling(n)`
  - JIT aliases such as `acb_mat_matmul_jit`, `acb_mat_solve_jit`, `acb_mat_qr_jit`
  - fixed/padded batch helpers such as `acb_mat_matmul_batch_fixed`, `acb_mat_matmul_batch_padded`, `acb_mat_det_batch_fixed`, `acb_mat_trace_batch_padded`

## Accuracy/Precision Semantics

- matrix entries are interpreted as complex boxes
- `matmul_basic`, `matvec_basic`, and `banded_matvec_basic` use box arithmetic directly
- `matvec_cached_prepare` / `matvec_cached_apply` reuse the validated box layout and currently share the same arithmetic path as `matvec_basic`
- `add`, `sub`, `neg`, `mul_entrywise`, and `conjugate` are exact box operations on the stored matrix entries
- `charpoly` is formed from midpoint eigenvalues; Hermitian midpoint structure uses the Hermitian eigensolver path
- `pow_ui` uses midpoint repeated squaring
- `exp` uses midpoint eigendecomposition and switches to the Hermitian path when midpoint Hermitian structure is detected
- `hermitian_part` is midpoint-based and exact for point boxes
- `is_hermitian` / `is_hpd` and the diagonal / triangular / real-valued predicates are midpoint structural diagnostics used to auto-route `solve` and `inv`
- `cho`, `ldl`, `hpd_solve`, and `hpd_inv` use midpoint Hermitian Cholesky on the Hermitian part, then outward boxing
- `eigvalsh` / `eigh` use midpoint Hermitian eigendecomposition on the Hermitian part, then outward boxing
- `solve` and `solve_basic` both currently use midpoint solve plus outward boxing
- `solve_tril`, `solve_triu`, and `solve_lu` are thin aliases over the triangular / LU plan paths so complex dense callers can keep Arb-style naming
- the dense factorization-solve ecosystem now also includes transpose solve, add-solve, and multi-RHS solve aliases on top of LU/HPD plan reuse
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
- the broader dense arithmetic / predicate / spectral / solve-alias surfaces now use the same fixed and padded batch conventions
- the matrix-function tranche (`charpoly`, `pow_ui`, `exp`) follows the same mode and padded-batch conventions
- direct dense kernels, dense `matvec`, and cached `matvec` remain separate first-class entry points; the factorization-solve ecosystem is additive on top of them rather than replacing them with operator objects
- generic `solve` / `inv` now auto-select the HPD route when midpoint Hermitian checks pass, so structured dense cases reuse the cheaper factorization path automatically

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.

## Current Gaps

- `solve_basic`, `inv_basic`, `triangular_solve_basic`, `lu_basic`, and `qr_basic` are still midpoint-first, not full complex-box enclosure algorithms
- rigorous coverage is strongest for `trace`, `det`, and the norm helpers; solve/factorization families do not yet have distinct tightened rigorous implementations
- large-`n` determinant bounds still fall back to midpoint determinant plus outward boxing
