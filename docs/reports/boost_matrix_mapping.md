# Boost Matrix Mapping

Last updated: 2026-03-09

This note maps locally available Boost matrix functionality, primarily
`boost::numeric::ublas`, against the canonical `arb_mat` / `acb_mat`
surface in this repo.

## Source Basis

- Boost headers inspected from `/usr/include/boost/numeric/ublas`
- Key headers:
  - `operation.hpp`
  - `operation_blocked.hpp`
  - `lu.hpp`
  - `triangular.hpp`
  - `matrix.hpp`
  - `matrix_expression.hpp`

## Important Framing

- Boost `ublas` is a dense/sparse linear-algebra container and algorithm layer.
- `arb_mat` / `acb_mat` are interval/box-valued JAX kernels with batching, AD,
  and precision-mode requirements.
- The useful comparison is capability-level, not one-to-one API imitation.

## Capability Map

| Boost / uBLAS capability | Evidence | `arb_mat` | `acb_mat` | Mapping note |
| --- | --- | --- | --- | --- |
| Dense matrix-matrix product | `prod`, `axpy_prod`, `block_prod` | yes | yes | Directly corresponds to `*_matmul` and `*_matmul_basic`. |
| Dense matrix-vector product | `prod`, `axpy_prod` | yes | yes | Directly corresponds to `*_matvec` and `*_matvec_basic`. |
| LU factorization | `lu_factorize` | yes | yes | Already present as `*_lu` / `*_lu_basic`. |
| LU solve / substitution | `lu_substitute` | partial | partial | Current surface has `solve`; no exposed LU-precomputed solve yet. |
| Triangular solve | `solve` / `inplace_solve` with triangular tags | yes | yes | Already present as `*_triangular_solve`. |
| Matrix inverse | typically built from LU + identity solve | yes | yes | Added here as canonical midpoint/outward-boxed `*_inv`. |
| QR factorization | not obvious in core inspected `ublas` headers | yes | yes | Added here as canonical midpoint/outward-boxed `*_qr`; not copied from Boost naming. |
| Permutation matrix support | `permutation_matrix` | partial | partial | Internal only via LU outputs; no dedicated public permutation helper type. |
| Identity / zero matrices | `identity_matrix`, `zero_matrix` | partial | partial | Used internally/conceptually; no public matrix constructors yet. |
| Structured matrix adaptors | triangular/hermitian adaptors | no | no | Potential future substrate once structure-aware kernels matter. |
| Matrix norms | `norm_1`, `norm_inf`, `norm_frobenius` | no | no | Reasonable next backlog item if we want Boost-like linear-algebra coverage. |
| Sparse/block optimized kernels | compressed/coordinated `axpy_prod`, blocked ops | no | no | Out of current dense interval/box scope. |

## Practical Backlog Interpretation

- Boost reinforces that the current canonical substrate should cover:
  - products
  - solves
  - factorizations
  - inverse
  - structural helpers
  - norms
- For this repo, the next high-value canonical additions after the current pass are:
  - LU-precomputed solve helpers
  - matrix norms
  - public identity/zero constructors
  - stronger rigorous enclosures for large-`n` determinant and inverse-like paths
- Boost sparse/block APIs should not be copied into `arb_mat` / `acb_mat`
  until there is an interval-aware sparse representation in the repo.

## Naming Guidance

- Keep Arb/FLINT-style canonical names in `arb_mat` / `acb_mat`.
- Use Boost only as an external capability reference.
- Do not introduce `ublas`-style container/adaptor names into the public JAX API.
