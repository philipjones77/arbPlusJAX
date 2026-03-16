Last updated: 2026-03-16T00:00:00Z

# Matrix Logdet and Matrix-Function Landscape

## Purpose

This note records the current large-scale matrix-function and log-determinant landscape relevant to arbPlusJAX and RandomFields77-style workloads.

The focus is practical:

- which methods are now standard in JAX-friendly scientific computing,
- what tradeoff each method makes,
- which parts of that landscape already exist in this repository,
- which parts are likely next additions rather than immediate dependencies.

## Core message

Modern large-scale kernel, Gaussian-process, and GMRF workflows no longer rely on a single dominant log-determinant strategy.

The current frontier is a tradeoff space between:

- Krylov / SLQ methods with correct reverse-mode gradients,
- variance-reduced stochastic trace estimators such as Hutch++ and related sketches,
- Leja-based polynomial approximations that reduce orthogonalisation cost,
- structured FFT embeddings for stationary grid problems,
- low-rank approximations and preconditioners such as pivoted Cholesky.

Different points on that frontier dominate under different geometry, sparsity, and gradient requirements.

## 1. Matrix-free Krylov with correct AD

For operator-first matrix functions, the most mature general-purpose route is still Krylov:

- Lanczos for symmetric / self-adjoint problems,
- Arnoldi for general operators.

The important recent shift is not merely forward evaluation; it is the availability of correct reverse-mode gradients through the Krylov procedure. That matters for:

- GP hyperparameter optimisation,
- GMRF precision learning,
- differentiable evidence approximations,
- matrix-function losses embedded inside JAX training loops.

In practice, this means that traces or quadratic forms of \(f(A)\) can be differentiated without naively backpropagating through every basis vector update in an unstable way. External JAX ecosystems such as `matfree` have made this direction more visible, and the same design pressure is already present in this repository.

Current repo status:

- `jrb_mat` implements matrix-free Lanczos actions and SLQ-style logdet with custom VJPs.
- `jcb_mat` implements the analogous Arnoldi path for complex/non-self-adjoint operators.
- `matfree_adjoints.py` contains a deeper adjoint-oriented linear-algebra substrate in the same spirit.

This remains the preferred path when:

- gradients must be correct and stable,
- the operator is only available through matvecs,
- the spectrum is not especially structured,
- orthogonalisation cost is acceptable.

## 2. Variance-reduced stochastic trace estimation

Plain Hutchinson estimators are often easy to implement but can be too noisy for production inference.

Variance-reduced methods such as:

- Hutch++,
- XTrace,
- XNysTrace,

improve the bias-variance-cost tradeoff by combining a low-rank sketch of the dominant subspace with residual stochastic probing.

For positive semidefinite or approximately low-effective-rank settings, this can reduce the number of probes required by a large factor relative to basic Hutchinson.

Why this matters here:

- log-likelihoods and evidence terms often require \(\operatorname{tr}(f(A))\) or \(\log \det(A)\),
- every probe is a full matrix-function action,
- reducing probe count directly reduces wall-clock cost.

Current repo status:

- the current `jrb_mat` stochastic logdet surface already supports Hutch++-style assembly through the Leja path.
- the broader trace-estimation family is not yet exposed as a general operator-sketch API.

This is likely the right direction for future unification:

- one operator-first trace-estimation chassis,
- multiple sketch policies layered on top,
- one diagnostics surface across Hutchinson, Hutch++, SLQ, and future variants.

## 3. Leja + Hutch++ for sparse SPD logdet

A March 2026 arXiv result introduces a particularly relevant alternative for sparse SPD log-determinant estimation [@Mbingui2026].

The idea is:

1. approximate \(\log(A) v\) over a spectral interval using Newton interpolation on real Leja points,
2. estimate \(\operatorname{tr}(\log(A))\) with Hutch++.

Relative to SLQ, this shifts work away from repeated orthogonalisation and toward polynomial action evaluation. That can be attractive for large sparse precision matrices where:

- orthogonalisation overhead is expensive,
- the operator is well-scaled on a known spectral interval,
- SPD structure is available.

Current repo status:

- `jrb_mat` now contains a sparse SPD Leja-log action route and a Hutch++-style logdet estimator,
- sparse `BCOO` Gershgorin bounds are used as a default spectral-bound starting point,
- the route is documented as an alternative to SLQ, not a replacement.

Practical interpretation:

- use SLQ when you want a familiar Krylov baseline with custom VJP support and tighter connection to the existing matrix-free chassis,
- use Leja + Hutch++ when orthogonalisation cost is the dominant bottleneck and a conservative spectral interval is available.

## 4. Toeplitz and circulant embedding for stationary kernels

On regular grids with stationary covariance structure, a more specialised family of methods often dominates both generic Krylov and generic trace estimation:

- Toeplitz structure,
- block Toeplitz structure,
- circulant embedding,
- FFT diagonalisation.

These can reduce sampling, solves, and often log-determinant-related work to quasi-linear complexity:

\[
O(n \log n).
\]

That advantage is highly problem-dependent:

- it is strongest on regular lattices,
- it depends on stationary covariance assumptions,
- it requires careful embedding design to avoid negative embedded eigenvalues,
- correctness can depend on cut-off, smoothing, or periodic-embedding choices.

For random-field workflows, periodic embedding and related structured approximations are already well established [@Guinness2019].

Current repo status:

- no first-class Toeplitz / circulant embedding logdet layer exists yet,
- this should be treated as a separate structured-matrix family, not folded blindly into the generic `jrb_mat` operator path.

This is likely the fastest route for:

- regular-grid stationary kernels,
- very large FFT-friendly lattice models,
- simulation-heavy RF77 workflows where the structure is explicit.

## 5. Low-rank approximations and preconditioners

Low-rank kernel approximations and preconditioners remain complementary rather than competing methods.

Pivoted Cholesky and related randomized low-rank constructions are useful because they can:

- compress rapidly decaying spectra,
- provide efficient approximate factorizations,
- act as preconditioners for CG, SLQ, or related trace/logdet estimators.

For smooth covariance kernels, this is often one of the highest-return engineering additions because it improves:

- conditioning,
- iteration counts,
- stochastic estimator variance,
- warm-start quality for repeated hyperparameter optimisation.

Current repo status:

- no first-class pivoted-Cholesky operator/preconditioner surface exists yet in the matrix-free path,
- but this is a natural complement to both the SLQ and Leja-based routes already present.
- sparse point solves now accept callable left preconditioners through the existing CG surface, so diagonal/Jacobi or other matrix-free preconditioner maps can already be inserted without changing the outer sparse solve API.

Practical AD note:

- callable JAX-native preconditioners are the safest immediate path because they stay inside the same traced JAX program,
- low-rank preconditioners are already conceptually aligned with the `matfree_adjoints.py` direction, but that module currently treats differentiation through the preconditioner itself conservatively rather than as a first-class public API,
- external pivoted-Cholesky utilities such as TensorFlow Probability's JAX substrate are therefore a design reference for future native implementations rather than a current dependency target.

## 6. Analytic VJPs, implicit adjoints, and preconditioned outer loops

For differentiable matrix functions, the right default remains:

- keep the outer matrix-function estimator inside JAX,
- differentiate with analytic VJPs or implicit adjoints,
- treat preconditioning as an accelerator for the linear algebra rather than as the main differentiation target.

This matches the existing repository direction:

- `jrb_mat` and `jcb_mat` expose custom-VJP matrix-free actions,
- preconditioners can be inserted in sparse solves,
- but the core AD contract is still attached to the matrix-function estimator, not to a black-box external sparse library.

That separation is important because it leaves room for future upgrades:

- better low-rank preconditioners,
- Ritz deflation,
- structured block preconditioners,
- optional external sparse backends at the matvec boundary only.

## 7. GPU sparse backends

Classic sparse preconditioners such as:

- ILU,
- IC(0),
- AMG,

remain highly relevant for very large PDE-style systems, especially on NVIDIA stacks. However, they should be viewed as a different integration tier.

In this repository, the clean architectural boundary would be:

- keep the JAX outer loop and estimator logic in-repo,
- move only the sparse operator/preconditioner application to an external backend when necessary,
- attach any custom VJP or numerical validation at that boundary instead of scattering external calls through the estimator internals.

That means these are plausible future directions, but not current default paths:

- cuSPARSE-backed sparse triangular / ILU kernels,
- AmgX-backed multigrid preconditioner calls,
- custom-call or callback bridges at the sparse operator boundary.

## 8. Practical hierarchy

The current practical hierarchy for this codebase is:

1. Toeplitz / circulant embedding:
   best when the geometry is regular-grid and stationary, and FFT structure is explicit.
2. SLQ / Lanczos with correct adjoints:
   best current general-purpose choice for differentiable matrix functions of symmetric operators.
3. Hutch++ and related variance-reduced estimators:
   best when probe variance dominates cost and a trace-estimation chassis is the main abstraction.
4. Leja + Hutch++:
   best as a sparse SPD alternative when orthogonalisation is too expensive and spectral bounds are available.
5. Pivoted Cholesky / low-rank preconditioners:
   best as a complement that improves almost all of the above.

The important engineering point is that these are modular:

- structured embedding is geometry-driven,
- Krylov is operator-driven,
- Hutch++ is estimator-driven,
- Leja is polynomial-approximation-driven,
- pivoted Cholesky is conditioning-driven.

They should therefore be integrated as interoperable first-class components rather than as one monolithic “logdet algorithm.”

## 9. Repository interpretation

The current arbPlusJAX direction is already consistent with that modular view:

- `jrb_mat` and `jcb_mat` are operator-first matrix-function surfaces,
- sparse `BCOO` support exists for large-scale JAX-native operators,
- SLQ/logdet and Leja/Hutch++ now coexist instead of being forced into a single path,
- custom-VJP matrix-free actions keep autodiff correctness central.

Likely next useful additions:

- a unified stochastic-trace API spanning Hutchinson, Hutch++, and future sketch variants,
- explicit preconditioner objects for matrix-free solvers and logdet estimators,
- lightweight Jacobi / block-diagonal preconditioner helpers for the sparse point layer,
- structured Toeplitz / circulant embedding modules for regular-grid random fields,
- benchmark suites that compare SLQ, Leja + Hutch++, and any future structured embedding route on the same sparse SPD workloads.

## 10. References

Bibliography keys currently available in the local library:

- `[@Mbingui2026]` for the Newton-Leja + Hutch++ sparse SPD logdet direction.
- `[@Guinness2019]` for periodic embedding / structured random-field spectral methodology.

External implementation ecosystems referenced here but not yet present as curated bibliography keys:

- `matfree`: https://github.com/pnkraemer/matfree
- `traceax`: https://github.com/mancusolab/traceax

Bibliography source used for key-based citations:

- `/mnt/c/dev/references/bibliography/library/mathematics.bib`
- `/mnt/c/dev/references/bibliography/library/random_fields.bib`
