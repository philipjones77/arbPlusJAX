Last updated: 2026-03-17T18:15:00Z

# Sparse Selected Inversion by Domain Decomposition

## Purpose

This note records the selected-inversion direction suggested by the `parsinv` work and how it should map into arbPlusJAX.

It is complementary to the current sparse log-determinant notes. The main targets here are:

- diagonal entries of `Q^{-1}`
- selected inverse entries of `Q^{-1}`
- trace terms such as `tr(Q^{-1} dQ)`

for large sparse SPD precision matrices.

## Problem setting

Let `Q` be a sparse SPD precision matrix. In latent Gaussian and GMRF workloads we often need:

- marginal variances: `diag(Q^{-1})`
- selected covariance entries
- hyperparameter derivatives involving `tr(Q^{-1} dQ)`

These are not the same target as `logdet(Q)`, even though both arise in the same inference pipelines.

## Reference direction

The paper "Parallel Selected Inversion for Space-Time Gaussian Markov Random Fields" proposes a parallel hybrid approach based on:

- sparse domain decomposition
- overlap regions between subdomains
- local direct solves on subdomain blocks
- stochastic correction for the interface error

In the upstream implementation, the global sparse problem is handled with PETSc `KSP`, overlap is managed through `PCGASM`, and the local direct solves use MUMPS-backed sparse Cholesky.

## Core algorithmic idea

The useful abstraction is:

1. partition the sparse graph into subdomains
2. enlarge each subdomain by an overlap width
3. compute local inverse information exactly on the overlapped subdomains
4. correct the missing global coupling with a stochastic estimator

This creates an accuracy/cost dial:

- more overlap increases local exactness
- less overlap is cheaper but leaves more work to the stochastic correction

That tradeoff is especially attractive for sparse precision matrices where:

- local neighborhoods are much smaller than the full system
- exact dense global inversion is impossible
- full sparse direct selected inversion does not scale well in distributed settings

## JAX translation

The arbPlusJAX runtime should preserve the idea, not the PETSc/MPI implementation.

The JAX-native translation is:

1. represent `Q` with repo-owned `SparseBCOO` or an equivalent operator closure
2. define a subdomain partition:
   - user-supplied blocks
   - or a simple graph partition heuristic
3. form overlap expansions by adjacency hops
4. extract each overlapped local block
5. compute local exact inverse data for that block
6. apply a matrix-free stochastic correction using the full sparse operator
7. aggregate:
   - inverse diagonal estimates
   - selected inverse entries
   - trace terms

## Parallelization model in JAX

The closest JAX-native analogue of the paper's parallelism is:

- `vmap` over subdomains
- `vmap` over stochastic probes
- `lax.scan` over iterative correction steps
- optional later sharding of batched subdomain/probe work through public JAX sharding APIs

This is different from MPI task orchestration, but it preserves the parallel structure that actually matters for accelerators and batched CPU execution.

The initial implementation should stay single-program and single-host. Multi-host distribution can come later if the JAX-native algorithm proves numerically useful.

## Recommended arbPlusJAX scope

The first stable sparse selected-inversion subset should be:

- `inverse_diagonal`
- `selected_inverse(pattern)`
- `trace_inv_dq`

with SPD point-mode semantics only.

That is the right minimal set because it directly serves:

- marginal variance estimation
- Laplace and GMRF hyperparameter gradients
- downstream covariance/precision diagnostics

## Proposed public shape

The natural home is `jrb_mat`, beside the current sparse logdet machinery.

Suggested additions:

- `jrb_mat_bcoo_inverse_diagonal_point(a, probes, overlap=..., partition=...)`
- `jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(...)`
- `jrb_mat_bcoo_selected_inverse_point(a, row_idx, col_idx, probes, overlap=..., partition=...)`
- `jrb_mat_bcoo_trace_inv_dq_point(a, dq, probes, overlap=..., partition=...)`

Current implemented subset:

- `jrb_mat_bcoo_inverse_diagonal_point(a, overlap=..., block_size=..., correction_probes=...)`
- `jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(...)`

The current implementation uses contiguous seed blocks, overlap expansion by graph hops, local dense inverse rows on each overlap block, and an optional stochastic diagonal correction
\[
\operatorname{diag}(A^{-1}) \approx \operatorname{diag}(B) + \frac{1}{m}\sum_{k=1}^m z_k \odot (x_k - B z_k), \qquad A x_k = z_k,
\]
where `B` is the row-wise local selected-inverse approximation built from the overlap blocks.

The diagnostics should report:

- partition count
- overlap width
- local block sizes
- correction probe count
- correction residual scale
- non-finite or ill-conditioned local-block flags

## Initial implementation advice

The first implementation should be intentionally modest:

- use user-supplied or trivial partitions first
- use dense local block inversion for extracted overlap blocks first
- use existing JAX-native CG for the global correction
- support only inverse diagonal first

That gives a workable correctness baseline before trying to build:

- better graph partitioning
- sparse local block factorizations
- selected off-diagonal extraction
- trace-gradient surfaces

## Relationship to current sparse logdet work

This method should coexist with the current sparse Leja plus Hutch++ logdet path.

The split of responsibilities is:

- Leja plus Hutch++: good for `logdet(Q)`
- selected inversion with overlap correction: good for `diag(Q^{-1})`, selected entries, and `tr(Q^{-1} dQ)`

They are complementary building blocks for the same inference workloads.

## Sources

- paper: <https://arxiv.org/abs/2309.05435>
- implementation reference: <https://github.com/abylayzhumekenov/parsinv>
