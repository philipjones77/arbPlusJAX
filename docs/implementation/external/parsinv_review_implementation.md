Last updated: 2026-03-17T18:15:00Z

# parsinv Review

This note records the initial audit of the `parsinv` repository as a design reference for sparse selected inversion in arbPlusJAX.

## Local checkout

- cloned to `/home/phili/projects/parsinv`
- audited commit: `3b2afcedcbaae0b85234855ccf3a7f544ca3dd9f`

## Upstream role

`parsinv` is a C/MPI project for:

- parallel selected inversion of sparse precision matrices
- marginal variance extraction
- trace terms for latent Gaussian model hyperparameter learning

The associated paper is:

- Abylay Zhumekenov, Elias T. Krainski, Håvard Rue
- "Parallel Selected Inversion for Space-Time Gaussian Markov Random Fields"
- arXiv `2309.05435`
- originally posted on September 11, 2023
- arXiv version 2 updated on March 16, 2026
- published in *Statistics and Computing* 35, 211 (2025)

## What the current code actually does

The implementation is built around PETSc and MPI, not around a pure sparse algebra kernel.

Important ingredients visible in the current source:

- global sparse solves through PETSc `KSP`
- domain-decomposition preconditioning through `PCGASM`
- subdomain overlap control
- local sparse direct Cholesky through MUMPS
- direct extraction of local inverse blocks
- stochastic correction of the overlap/interface error

Concrete examples:

- `ParsinvInverseKSPCreate(...)` configures `PCGASM` with overlap and MUMPS-backed local Cholesky
- `ParsinvInverseMatInvert(...)` extracts the local inverse from the MUMPS factor object
- `ParsinvInverseMatCorrect(...)` applies the stochastic correction loop
- `ParsinvInverseMatMatTraceSparseSeq(...)` forms sparse Hadamard products to accumulate trace terms

## Why this matters for arbPlusJAX

This is relevant to arbPlusJAX, but not as a literal code-port target.

The key reusable ideas are:

- overlap-based sparse domain decomposition
- exact local selected-inverse blocks on subdomains
- stochastic correction for interface error
- direct estimation of:
  - diagonal entries of `Q^{-1}`
  - selected inverse entries
  - trace terms of the form `tr(Q^{-1} dQ)`

Those are valuable because they target a different quantity family than the current sparse Leja plus Hutch++ logdet route.

## What should not be copied directly

The current `parsinv` stack depends on:

- MPI
- PETSc `KSP` / `PCGASM`
- MUMPS

That is not the right runtime substrate for canonical arbPlusJAX code.

For arbPlusJAX:

- the runtime surface should remain JAX-native
- PETSc/MPI/MUMPS can be used only as optional external references or benchmark baselines
- differentiation contracts should remain on the JAX side

## Recommended translation into arbPlusJAX

The right interpretation is:

- `parsinv` is a design reference for a new sparse inverse-diagonal / selected-inverse estimator family
- it should live beside the current `jrb_mat` sparse logdet methods, not replace them

The first JAX-native target should therefore be:

1. sparse SPD inverse-diagonal estimation
2. sparse selected inverse on user-specified index patterns
3. trace estimators for `tr(Q^{-1} dQ)`

not:

- a direct MPI port
- a PETSc wrapper as the canonical runtime path

## Proposed JAX mapping

The rough translation is:

- sparse precision matrix stored as repo-owned `SparseBCOO`
- graph partition or user-supplied block partition
- overlap expansion by adjacency hops
- local subdomain extraction to dense blocks for initial implementation
- local exact solves/inverses in JAX for those blocks
- global correction using matrix-free CG or another JAX-native solver
- probe batching through `vmap`
- correction loops through `lax.scan`

This keeps:

- the algorithmic idea
- the selected-inversion target
- the overlap-accuracy tradeoff

while dropping:

- the PETSc runtime dependency
- the MPI-specific control flow

## Relationship to current repo status

arbPlusJAX already has:

- sparse point matrices in `srb_mat` / `scb_mat`
- sparse operator closures and spectral helpers in `jrb_mat`
- matrix-free CG hooks
- sparse SPD logdet through Leja plus Hutch++
- a first JAX-native sparse inverse-diagonal estimator built from overlap-domain local inverse rows plus optional stochastic correction

arbPlusJAX does not yet have:

- a selected-inverse estimator on sparse patterns
- a `tr(Q^{-1} dQ)` sparse estimator family

So this is a real gap, and `parsinv` is a strong design reference for filling it.

## Sources

- `parsinv` repository: <https://github.com/abylayzhumekenov/parsinv>
- paper: <https://arxiv.org/abs/2309.05435>
