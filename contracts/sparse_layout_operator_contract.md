Last updated: 2026-03-29T00:00:00Z

# Sparse Layout And Operator Contract

## Scope

This contract covers the current repo-owned sparse containers, sparse matrix APIs, and matrix-free operator expectations used by the sparse real and sparse complex layers.

## Repo-owned sparse substrate

Runtime sparse code under `src/arbplusjax/` must use the repo-owned sparse substrate in `arbplusjax.sparse_common`.

Current core container families include:

- `SparseCOO`
- `SparseCSR`
- `SparseBCOO`
- `SparseIntervalCOO`
- `SparseIntervalCSR`
- `SparseIntervalBCOO`
- `SparseBoxCOO`
- `SparseBoxCSR`
- `SparseBoxBCOO`
- `SparseMatvecPlan`
- `SparseQRFactor`

The block and variable-block sparse container families in the same module follow the same repo-owned policy.

## Storage and conversion contract

- Sparse real and sparse complex matrix APIs support conversion to and from dense arrays.
- COO, CSR, and BCOO are current supported storage families for the main sparse real and sparse complex surfaces.
- Block-sparse and variable-block sparse families are part of the sparse storage
  contract when they expose analogous public surfaces.
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

## Sparse-Native Operational Contract

The sparse operational apply surface is explicitly governed.

For `srb_mat` and `scb_mat`, the following operational routes are contractually
distinct from dense-lifted sparse helpers:

- point `matvec`
- point `rmatvec`
- point cached `prepare/apply`
- basic `matvec`
- basic `rmatvec`
- basic cached `prepare/apply`
- compiled repeated-call binders over cached sparse apply

For those routes:

- sparse operational execution must not silently densify
- dense fallback is not allowed unless the contract is updated explicitly
- CPU/GPU operational benchmarking should measure these routes directly rather
  than infer behavior from broader solve/factor benchmarks

This contract does not say that every sparse `basic` solve/factor route is
sparse-native today. It only freezes the sparse-native operational apply layer
as a separate guarantee boundary.

## Correctness and non-goals

- Dense roundtrip, transpose/conjugate-transpose, matvec, solve, LU, QR, and norm behavior are part of the current sparse conformance surface.
- Exact storage-specific performance is not frozen by this contract.
- Sparse-sparse multiplication internals may change, including temporary dense fallback strategies, as long as the public numerical result contract is preserved.
- Sparse solve/factor internals may still use dense lifting on non-operational
  `basic` paths where that limitation is documented; callers must not infer from
  this contract that every sparse `basic` route is already sparse-native.

## Source of truth

- `src/arbplusjax/sparse_common.py`
- `src/arbplusjax/srb_mat.py`
- `src/arbplusjax/scb_mat.py`
- `tests/test_sparse_operational_contracts.py`
- `docs/implementation/modules/jrb_mat_implementation.md`
- `docs/implementation/modules/jcb_mat_implementation.md`
- `tests/test_srb_mat_chassis.py`
- `tests/test_scb_mat_chassis.py`
- `tests/test_jrb_mat_chassis.py`
- `tests/test_jrb_mat_logdet_contracts.py`
- `tests/test_jcb_mat_chassis.py`
