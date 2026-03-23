Last updated: 2026-03-17T00:00:00Z

# Sparse Layout And Operator Contract

## Scope

This contract covers the current repo-owned sparse containers, sparse matrix APIs, and matrix-free operator expectations used by the sparse real and sparse complex layers.

## Repo-owned sparse substrate

Runtime sparse code under `src/arbplusjax/` must use the repo-owned sparse substrate in `arbplusjax.sparse_common`.

Current core container families include:

- `SparseCOO`
- `SparseCSR`
- `SparseBCOO`
- `SparseMatvecPlan`
- `SparseQRFactor`

The block and variable-block sparse container families in the same module follow the same repo-owned policy.

## Storage and conversion contract

- Sparse real and sparse complex matrix APIs support conversion to and from dense arrays.
- COO, CSR, and BCOO are current supported storage families for the main sparse real and sparse complex surfaces.
- Sparse containers are JAX pytrees and must remain usable as traced payloads in repo-owned JAX workflows.
- The runtime sparse surface must not depend on `jax.experimental.sparse`.

## Layout and shape contract

- Sparse matrices expose logical `(rows, cols)` shape metadata.
- Sparse vectors and right-hand sides use dense JAX arrays at the operator boundary unless a more specific helper explicitly states otherwise.
- Real operator-style interval matrices use the `jrb_mat` canonical interval layouts.
- Complex operator-style box matrices use the `jcb_mat` canonical box layouts.

## Operator contract

- Matrix-free operator callbacks must be pure-JAX and fixed-shape.
- Real symmetric matrix-free paths may reuse the same operator as the adjoint.
- Complex non-self-adjoint matrix-free paths require explicit adjoint threading where the API requests `adjoint_matvec`.
- Sparse cached matvec plans may change internal representation, but must preserve matvec semantics.

## Correctness and non-goals

- Dense roundtrip, transpose/conjugate-transpose, matvec, solve, LU, QR, and norm behavior are part of the current sparse conformance surface.
- Exact storage-specific performance is not frozen by this contract.
- Sparse-sparse multiplication internals may change, including temporary dense fallback strategies, as long as the public numerical result contract is preserved.

## Source of truth

- `src/arbplusjax/sparse_common.py`
- `src/arbplusjax/srb_mat.py`
- `src/arbplusjax/scb_mat.py`
- `docs/implementation/modules/jrb_mat_implementation.md`
- `docs/implementation/modules/jcb_mat_implementation.md`
- `tests/test_srb_mat_chassis.py`
- `tests/test_scb_mat_chassis.py`
- `tests/test_jrb_mat_chassis.py`
- `tests/test_jrb_mat_logdet_contracts.py`
- `tests/test_jcb_mat_chassis.py`

