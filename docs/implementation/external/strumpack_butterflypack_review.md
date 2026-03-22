Last updated: 2026-03-21T09:59:09Z

# STRUMPACK / ButterflyPACK Review

This note records the current STRUMPACK / ButterflyPACK ecosystem as an external implementation lineage relevant to arbPlusJAX matrix benchmarks, hierarchical factorization planning, and differentiable logdet design.

## Scope

The focus here is narrow:

- what STRUMPACK and ButterflyPACK provide today as practical hierarchical solver infrastructure
- how PETSc and SLEPc expose that stack to Python-facing workflows
- what recent randomized hierarchical approximation results change in the theoretical landscape
- what still appears to be missing for first-class JAX-native autodiff integration

This is not a proposal to make STRUMPACK or ButterflyPACK canonical runtime dependencies.

## What the current ecosystem provides

Current upstream documentation shows that STRUMPACK is a mature hierarchical solver package rather than a single-format direct solver.

From the current STRUMPACK dense-solver documentation and current release notes:

- STRUMPACK exposes dense rank-structured formats including HSS, BLR, HODLR, Butterfly, and HODBF
- the `StructuredMatrix` interface supports construction from:
  - dense matrices
  - matrix-vector products
  - selected entries
- the same interface exposes:
  - matrix-vector products
  - transpose products
  - factorization and solve
  - memory/rank inspection
- recent upstream releases continue to advertise GPU-oriented sparse-solver work and GPU performance improvements, including an experimental CUDA-only sparse symmetric solver in `v8.0.0`

In PETSc, the current `MATSOLVERSTRUMPACK` interface makes the solver usable through PETSc factorization paths and exposes current compression and GPU options.

The current PETSc manual pages show:

- STRUMPACK is available as a PETSc matrix solver backend via `MATSOLVERSTRUMPACK`
- low-rank compression modes exposed through PETSc include HSS, BLR, HODLR, BLR_HODLR, and ZFP-based hybrids
- ButterflyPACK integration is required for HODLR / HODBF-related compression paths
- PETSc exposes `-mat_strumpack_gpu` for GPU acceleration in numerical factorization, with an explicit note that it is not supported for all compression types
- PETSc recommends `1 MPI process per GPU`

So the present practical picture is:

- STRUMPACK and ButterflyPACK are active, real solver backends in the PETSc ecosystem
- they already matter for sparse approximate factorization and rank-structured compression benchmarking
- they are still solver-stack components in C++ / Fortran / MPI ecosystems, not JAX primitives

## Python-facing integration story

The Python access path is also clear in the current docs:

- PETSc has Python bindings via `petsc4py`
- SLEPc has Python bindings via `slepc4py`
- SLEPc relies on PETSc data structures and linear solvers
- `slepc4py` must be used together with `petsc4py`

That means Python can drive these ecosystems today, but through PETSc/SLEPc object models rather than through XLA-native primitives.

For arbPlusJAX, the practical value is:

- benchmark baselines
- external solver comparisons
- operator/eigensolver references

not:

- native traced JAX arrays with built-in autodiff rules
- first-class JIT-fusible hierarchical factorization kernels

## Randomized hierarchical approximation results

Recent theory materially improves the picture for hierarchical compression quality control.

The 2024 HODLR result of Chen, Keles, Halikias, Musco, Musco, and Persson gives a near-optimal randomized HODLR approximation algorithm in the matrix-vector query model.

The current abstract states that:

- the method works from matrix-vector products with `A` and `A^T`
- it provides worst-case approximation guarantees for HODLR approximation
- it gives explicit query complexity bounds
- the analysis includes stability arguments for noisy low-rank approximation and level-to-level error propagation

The 2025 follow-up on HSS gives a quasi-optimal randomized HSS approximation result with matrix-vector access and explicit error guarantees.

The current abstract states that:

- the algorithm produces an HSS approximation using only matrix-vector products with `A` and `A^T`
- it obtains quasi-optimal Frobenius-error guarantees up to logarithmic factors
- it is, according to the abstract, the first polynomial-time quasi-optimality result for HSS approximation in this model

Why this matters for us:

- hierarchical compression can now be discussed with sharper approximation guarantees instead of only empirical folklore
- sketch quality, tolerance selection, and rank adaptivity can be treated as first-class numerical controls
- this strengthens the case for benchmarking hierarchical approximations against stochastic logdet and matrix-free baselines

## What still appears to be missing for JAX

Engineering inference from the cited STRUMPACK, PETSc, SLEPc, and JAX materials:

- there is still no de facto standard JAX-native hierarchical factorization package with first-class `jax.custom_vjp`-style reverse rules for hierarchical solves, eigensolvers, or logdet pipelines
- the current production-ready hierarchical solver stacks remain external runtimes with Python wrappers, not XLA-lowered JAX operators
- using them inside JAX would still require explicit glue such as:
  - callback-based wrappers
  - a custom primitive / `custom_call`
  - `jax.custom_vjp` or related custom derivative machinery around implicit backward solves

This inference is consistent with the current JAX documentation:

- `jax.pure_callback` can participate in `jit` and `vmap`, but autodiff is not defined unless the user supplies custom derivative rules
- `jax.custom_vjp` is the standard mechanism for defining reverse-mode rules when JAX should not differentiate through the wrapped implementation directly

So the missing piece is not merely "a Python binding exists." The missing piece is:

- a stable JAX-visible primitive boundary
- a correct backward rule that reuses the same compressed operator semantics
- disciplined handling of approximation metadata across forward and backward passes

## Practical implications for arbPlusJAX

For this repo, the STRUMPACK / ButterflyPACK ecosystem should be treated as:

- benchmark and comparison infrastructure
- an external implementation lineage for hierarchical sparse/dense solvers
- a design reference for compressed-factor workflows

It should not yet be treated as:

- a canonical JAX runtime substrate
- a drop-in differentiable logdet backend

Near-term implications:

1. Keep STRUMPACK / PETSc / SLEPc in the benchmark and external-implementation layer.
2. Compare them against:
   - repo-native sparse and matrix-free paths
   - SciPy and `jax.experimental.sparse`
   - stochastic SLQ / Leja / selected-inverse style estimators
3. Treat gradient-correct hierarchical logdet as a separate engineering task requiring an explicit JAX integration boundary.

Numerical caution:

- if forward compression tolerance, rank truncation, or butterfly level selection are allowed to drift independently from the backward pass, gradient estimates can become biased or noisy
- if we ever build a JAX wrapper, the backward rule should reuse the same compressed operator or an explicitly controlled adjoint approximation

## Recommended repo interpretation

The right interpretation for arbPlusJAX is:

- STRUMPACK / ButterflyPACK are important external baselines for hierarchical matrix methods
- current PETSc / SLEPc bindings make them operationally reachable from Python
- current JAX integration remains custom engineering, not an existing library feature
- recent randomized HODLR/HSS theory makes hierarchical approximation more defensible as a benchmarked method family

That puts them in the same bucket as other external implementation reviews in this repo:

- worth tracking
- worth benchmarking
- not yet a reason to move the canonical runtime away from JAX-native implementations

## Sources

- STRUMPACK dense-solver docs: <https://portal.nersc.gov/project/sparse/strumpack/master/dense.html>
- STRUMPACK releases: <https://github.com/pghysels/STRUMPACK/releases>
- PETSc `MATSOLVERSTRUMPACK`: <https://petsc.org/release/manualpages/Mat/MATSOLVERSTRUMPACK/>
- PETSc `MatSTRUMPACKCompressionType`: <https://petsc.org/main/manualpages/Mat/MatSTRUMPACKCompressionType/>
- PETSc overview / petsc4py entrypoint: <https://petsc.org/release/>
- petsc4py docs: <https://petsc.org/release/petsc4py/>
- SLEPc overview: <https://slepc.upv.es/release/index.html>
- slepc4py docs: <https://slepc.upv.es/release/slepc4py/>
- Chen et al., "Near-optimal hierarchical matrix approximation from matrix-vector products" (arXiv 2407.04686): <https://arxiv.org/abs/2407.04686>
- Amsel et al., "Quasi-optimal hierarchically semi-separable matrix approximation" (arXiv 2505.16937): <https://arxiv.org/abs/2505.16937>
- JAX external callbacks: <https://docs.jax.dev/en/latest/external-callbacks.html>
- JAX `custom_vjp`: <https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html>
