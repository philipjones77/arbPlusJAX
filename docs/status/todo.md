Last updated: 2026-03-27T00:00:00Z

# TODO

This is the top-level implementation TODO index for active repo work in
`docs/status/`.

Detailed cross-cutting and specialized backlogs are split into dedicated files:
- [cross_cutting_todo.md](/docs/status/cross_cutting_todo.md)
- [theory_todo.md](/docs/status/theory_todo.md)
- [curvature_todo.md](/docs/status/curvature_todo.md)
- [special_functions_todo.md](/docs/status/special_functions_todo.md)
- [api_runtime_todo.md](/docs/status/api_runtime_todo.md)
- [release_packaging_todo.md](/docs/status/release_packaging_todo.md)
- [docs_publishing_todo.md](/docs/status/docs_publishing_todo.md)
- [security_supply_chain_todo.md](/docs/status/security_supply_chain_todo.md)
- [operational_support_todo.md](/docs/status/operational_support_todo.md)
- [capability_maturity_todo.md](/docs/status/capability_maturity_todo.md)
- [production_readiness_todo.md](/docs/status/production_readiness_todo.md)

Organization rule:
- Sections follow the top-level function and infrastructure categories defined in
  [test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).
- Test-owner priorities follow
  [test_gap_checklist.md](/docs/status/test_gap_checklist.md).
- Generated inventories and detailed provenance stay in `docs/reports/`,
  not here.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

Current phase snapshot:
- Tier 0 architecture/API: `in_progress`
- point-fast JAX conversion of public point mode across all six categories: `done`
- Tier 1 core special-function hardening: `in_progress`
- Tier 2 general incomplete-tail engine: `in_progress`
- Tier 3 incomplete Bessel specialization: `in_progress`
- Tier 4 higher extensions: `in_progress`
- multivariate Bessel placement after scalar incomplete infrastructure: `planned`

## Cross-cutting execution and run-platform

Status: `in_progress`

Detailed backlog:
- [cross_cutting_todo.md](/docs/status/cross_cutting_todo.md)

## Cross-cutting point-fast JAX conversion

Status: `in_progress`

Detailed backlog:
- [cross_cutting_todo.md](/docs/status/cross_cutting_todo.md)
- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)

## Cross-repo provider boundary

Status: `in_progress`

Detailed backlog:
- [cross_cutting_todo.md](/docs/status/cross_cutting_todo.md)

## 1. Core Numeric Scalars

Status: `done`

- `done`
  - direct test owners exist for the main scalar chassis:
    `arb_core`, `acb_core`, `arf`, `acf`, `fmpr`, `fmpzi`, `arb_fpwrap`
  - direct test owners also exist for supporting scalar layers:
    `arb_calc`, `acb_calc`, `mag`, `fmpz_extras`
  - representative public-API coverage now exists for scalar helper families:
    `arf`, `acf`, `fmpr`, `fmpzi`, and `arb_fpwrap`
  - the existing scalar performance scripts:
    `benchmark_arf.py`, `benchmark_acf.py`, `benchmark_fmpr.py`,
    `benchmark_fmpzi.py`, and `benchmark_arb_fpwrap.py`
    now emit shared benchmark-report JSON with cold/warm/recompile timing
  - the existing core comparison scripts:
    `compare_arb_core.py` and `compare_acb_core.py`
    now emit shared benchmark-report JSON for scalar accuracy artifacts
- `in_progress`
  - none
- `done`
  - CPU direct-owner and parity slices now run against the repo-local WSL C
    reference builds through the shared parity-path helper
  - direct scalar status rollup now exists in
    [core_numeric_scalars_status.md](/docs/reports/core_numeric_scalars_status.md),
    including the distinction between true interval/box scalar kernels
    (`arb_core`, `acb_core`) and point-only helper scalar families
    (`arf`, `acf`, `fmpr`, `fmpzi`, `arb_fpwrap`)
  - canonical CPU notebook outputs now exist for the scalar/API tranche under
    [example_core_scalar_surface](/examples/outputs/example_core_scalar_surface)
    and
    [example_api_surface](/examples/outputs/example_api_surface)
  - scalar/API notebooks now explicitly show production service usage with
    binder reuse, stable dtype policy, and optional padding/chunking
  - the owned scalar CPU parity/owner slice now passes end-to-end, and the
    owned scalar GPU JAX-facing slice also passes on CUDA
  - retained scalar benchmark and notebook artifacts now exist for both CPU and
    GPU policy runs, and the backend-realized conclusion is explicit: CPU
    remains the default winner for many tiny scalar service workloads while GPU
    is validated for larger repeated scalar batches

## 2. Interval / Box / Precision Modes

Status: `in_progress`

- `done`
  - direct test owners exist for `double_interval`, `mp_mode`, and
    `jax_precision`
  - a direct owner test now exists for
    [test_point_wrappers_contracts.py](/tests/test_point_wrappers_contracts.py),
    covering representative point-wrapper exports, batching, padding, matrix
    plan prepare/apply paths, and API fastpath parity
  - mode tests exist for point/basic interval behavior and matrix mode routing
  - targeted helper coverage now exists for shared batch padding, trimming,
    midpoint conversion, and low-level shape guards in the existing runtime/API
    test surface
  - the public interval repeated-call API now has the same backend-aware
    policy and diagnostics surface as point mode through
    `choose_interval_batch_policy(...)`,
    `bind_interval_batch_with_diagnostics(...)`,
    `bind_interval_batch_jit(...)`,
    `bind_interval_batch_jit_with_diagnostics(...)`, and
    `prewarm_interval_mode_kernels(...)`
  - direct owner tests now cover the new interval service API surface on CPU
    and CUDA in
    [test_interval_mode_service_contracts.py](/tests/test_interval_mode_service_contracts.py)
  - the routed API benchmark now records interval service binder versus direct
    padded interval batch timing on both CPU and GPU in
    [benchmark_api_surface_cpu_modes_refresh.json](/benchmarks/results/benchmark_api_surface/benchmark_api_surface_cpu_modes_refresh.json)
    and
    [benchmark_api_surface_gpu_modes_refresh.json](/benchmarks/results/benchmark_api_surface/benchmark_api_surface_gpu_modes_refresh.json)
  - the canonical routed API notebook source now teaches the interval
    diagnostics/policy layer in [example_api_surface.ipynb](/examples/example_api_surface.ipynb)
- `in_progress`
  - continue tightening wrapper ownership for the remaining indirect wrapper
    modules beyond the now-landed `hypgeom_wrappers`, `poly_wrappers`,
    `dft_wrappers`, `dirichlet_wrappers`, and `modular_elliptic_wrappers`
  - direct owner tests now exist for `ball_wrappers`,
    `baseline_wrappers`, and `mat_wrappers`
  - direct owner tests now exist for `wrappers_common`, `core_wrappers`,
    `calc_wrappers`, and `double_interval_wrappers`
  - direct owner tests now exist for `hypgeom_wrappers`, `poly_wrappers`,
    `dft_wrappers`, `dirichlet_wrappers`, and `modular_elliptic_wrappers`
  - direct owner tests now exist for `checks`, `coeffs`, and `precision`
  - keep precision-routing and dtype policy explicit instead of letting wrapper
    behavior drift by family
  - finish the retained CPU/GPU executed-notebook refresh for the updated
    interval/routed API notebook outputs under
    [examples/outputs/example_api_surface](/examples/outputs/example_api_surface)
- `planned`
  - add a repo-wide wrapper contract matrix covering shape normalization,
    broadcasting, dtype promotion, batch padding, and mode dispatch invariants

## 3. Dense Matrix Functionality

Status: `done`

- `done`
  - dense `arb_mat` / `acb_mat` now cover the main public chassis:
    `zero`, `identity`, `permutation_matrix`, `transpose`,
    `conjugate_transpose`, `submatrix`, `diag`, `diag_matrix`, `matmul`,
    `matvec`, `banded_matvec`, cached matvec prepare/apply, dense plan helpers,
    `solve`, `inv`, `sqr`, `det`, `trace`, `norm_fro`, `norm_1`, `norm_inf`,
    `triangular_solve`, `lu`, and `qr`
  - point/basic/adaptive/rigorous API exposure exists for the main chassis
  - dense benchmark and example coverage exist in
    [benchmark_dense_matrix_surface.py](/benchmarks/benchmark_dense_matrix_surface.py),
    [dense_matrix_surface_benchmark.md](/docs/reports/dense_matrix_surface_benchmark.md),
    and
    [example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb)
- `done`
  - exact-reference dense chassis checks now cover `inv`, `qr`, cached matvec,
    determinant, trace, and matrix norms for real and complex dense matrices
  - dense chassis coverage now also includes point/basic/JIT/API parity checks
    for banded matvec on real and complex paths
  - large-`n` determinant rigor now uses midpoint-perturbation / Hadamard
    plus cofactor-Lipschitz enclosures instead of aliasing `basic`, with
    explicit 5x5/6x6 reference-nesting checks on both real and complex paths
  - separate midpoint-first solve-family implementations from true interval/box
    linear-algebra kernels in status and engineering reports
  - rigorous SPD/HPD, LU, and generic dense solve/inverse paths now use
    residual-enclosed midpoint solves
  - rigorous factor outputs now have widened `cho` / `ldl` / `lu` / `qr`
    surfaces instead of returning only midpoint factors, while preserving
    unit-diagonal and triangular zero structure in the widened outputs
- `planned`
  - generate dense engineering status from the capability registry instead of
    maintaining [dense_matrix_engineering_status.md](/docs/reports/dense_matrix_engineering_status.md)

## 4. Sparse / Block-Sparse / VBlock Functionality

Status: `in_progress`

- `done`
  - sparse `srb_mat` / `scb_mat` point-mode layers exist with `COO`, `CSR`,
    `BCOO`, sparse matvec, cached matvec, batching, sparse algebra, iterative
    solves, pivoted LU, and structured Householder QR
  - fixed-block `srb_block_mat` / `scb_block_mat` point-mode surfaces exist for
    BSR-like storage, matvec/rmatvec, cached apply, batch helpers, triangular
    solve, LU wrappers, and QR wrappers
  - variable-block `srb_vblock_mat` / `scb_vblock_mat` surfaces exist for
    partitioned COO/CSR, dense conversion, apply, cached apply, matmul, and
    direct solve/factorization helpers
  - variable-block direct factorization/solve helpers now support square dense
    matrices even when row/column partition vectors differ, instead of
    requiring hidden square-partition layouts
  - for the current sparse/block/vblock tranche, hardening has been completed
    for `point` and `basic` coverage rather than pretending adaptive/rigorous
    sparse modes were finished in the same pass
  - the retained sparse CPU validation profile now runs through
    [tools/run_test_harness.py](/tools/run_test_harness.py) and is recorded in
    [cpu_validation_profiles.md](/docs/reports/cpu_validation_profiles.md)
  - the sparse, block-sparse, and variable-block benchmark/report layer now
    follows the shared schema with pytest-owned runtime-report checks
  - direct-owner tests now exist for `block_sparse_core` and `sparse_core`
  - wrapper-level mode-dispatch surfaces now also exist for block-sparse and
    variable-block sparse families through
    [mat_wrappers.py](/src/arbplusjax/mat_wrappers.py), with direct owner tests
    in
    [test_block_vblock_sparse_mode_surface.py](/tests/test_block_vblock_sparse_mode_surface.py)
- `in_progress`
  - direct owner tests now exist for `sparse_common`
- `planned`
  - extend native interval/box sparse coverage beyond the now-landed direct core
    `basic` determinant, inverse, square, factor, and solve entrypoints into
    more of the remaining sparse algorithm surface; main-stack sparse `basic`
    LU/Cholesky plan preparation should stay on dense interval/box plans rather
    than reusing sparse point plans
  - extend the storage-format preparation versus solver-quality benchmark split
    beyond the main sparse and current block/vblock reports into additional
    sparse benchmark families; the main sparse surface now tracks
    `storage_prepare`, `cached_prepare`, and LU / SPD / HPD factor-plan
    preparation separately from solve and apply timings

## 5. Matrix-Free / Operator Functionality

Status: `in_progress`

### Current Landed Surface

- `done`
  - the earlier matrix-free import/startup stall has been fixed structurally:
    eager module-level JIT alias blocks in `jrb_mat` / `jcb_mat` were replaced
    by lazy wrappers, and the same shared lazy-JIT helper now also covers
    `arb_core`, `nufft`, `dirichlet`, and `acb_dirichlet`
  - `acb_core` no longer imports `hypgeom`, `barnesg`, or the Dirichlet family
    at module import time; those heavy dependencies are now deferred to the
    specific call sites that need them
  - `acb_core` now uses a lazy decorator pattern for its JIT-decorated public
    functions instead of paying decorator-time wrapper construction during
    import
  - `acb_dirichlet` now defers `series_missing_impl` fallback loading behind
    `__getattr__` rather than importing that fallback namespace eagerly
  - startup-compile regression coverage now exists in
    [test_startup_compile_policy.py](/tests/test_startup_compile_policy.py)
  - `jrb_mat` and `jcb_mat` now provide matrix-free Krylov action, trace
    estimator, and SLQ-style logdet scaffolding with targeted tests and
    benchmark coverage
  - custom VJP support exists for key action, quadratic-form, and trace-facing
    paths
  - restarted and block-RHS `expm` action variants exist on the Jones paths
  - operator-plan surfaces now cover shell, finite-difference, sparse `BCOO`,
    block-sparse, and variable-block sparse adapters
  - reusable midpoint Krylov solve scaffolding now lives in `matrix_free_core`
    rather than duplicated wrappers
  - direct owner tests now exist for `matrix_free_core`
  - direct owner tests now exist for `krylov_solvers`
  - direct owner tests now exist for `kernel_helpers`, `mat_common`, and
    `sampling_helpers`
  - eigensolver diagnostics surfaces exist across Lanczos, Arnoldi, restarted,
    block, Krylov-Schur, Davidson, Jacobi-Davidson, shift-invert, and contour
    entry points
  - `matrix_free_core` now includes reusable contour-integral action helpers,
    rational/polynomial spectral-action helpers, and a shared operator-first
    `logdet_solve` combiner used by the real/complex Jones wrappers
  - public matrix-free rational spectral-action surfaces now exist on
    `jrb_mat` / `jcb_mat`, backed by the shared core helper substrate and
    diagonal-case chassis proofs
  - orthogonal probe-block generation now exists in `matrix_free_core` with
    public `jrb_mat` / `jcb_mat` wrappers for real and complex probe blocks
  - reusable pilot-variance and adaptive probe-budget helpers now exist in
    `matrix_free_core`, with public trace-estimator probe statistics/adaptive
    count surfaces on `jrb_mat` / `jcb_mat`
  - shared Hutch++ and SLQ block-action/metadata-preparation logic now lives in
    `matrix_free_core`, with the real/complex Jones wrappers reduced to
    algebra-specific coercion and midpoint policy
  - SLQ/Hutch++ postprocessing now lives in shared `matrix_free_core` helpers
    rather than duplicated real/complex wrapper code
  - the public estimator aliases now line up more cleanly across `jrb_mat` and
    `jcb_mat`, including `trace_estimate_*`, `logdet_estimate_*`, and heat-trace
    basic surfaces
  - shared probe-statistics stopping helpers and shared correction-expansion
    helpers now live in `matrix_free_core` instead of duplicated real/complex
    wrapper logic
  - public contour-integral `log`, `sqrt`, `root`, and `sign` action wrappers
    now exist on `jrb_mat` / `jcb_mat`, backed by the shared contour-integral
    helper substrate
  - public contour-integral `sin` / `cos` action wrappers now also exist on
    `jrb_mat` / `jcb_mat`
  - public contour-integral `sinh` / `cosh` / `tanh` action wrappers now also
    exist on `jrb_mat` / `jcb_mat`
  - public contour-integral `exp` / `tan` action wrappers now also exist on
    `jrb_mat` / `jcb_mat`
  - shared eigensolver convergence accounting now lives in
    `matrix_free_core.eig_convergence_summary` instead of duplicated
    real/complex residual-threshold bookkeeping
  - low-rank deflation metadata and residual trace-estimation helpers now live
    in shared `matrix_free_core`, with public real/complex Jones wrappers for
    preparing and reusing deflated operator state across probe passes
  - cached rational Hutch++ metadata now exists for the real/complex Jones
    wrappers, so rational trace and rational-logdet approximants can reuse the
    same compact deflation state instead of rebuilding it on every pass
  - cached rational Hutch++ metadata now also records cached-adjoint support
    and transpose-preconditioner reuse when the chosen structure/policy can
    support that path
  - shared probe-budget helpers now expose probe deficit and next total probe
    count contracts in addition to the existing target-met / should-stop tests
  - Davidson and Jacobi-Davidson now share restart target-column policy through
    `matrix_free_core`, and locked residual corrections are filtered before
    basis expansion
  - operator-first solve surfaces on `jrb_mat` / `jcb_mat` now route through
    `matrix_free_core.implicit_krylov_solve_midpoint` using
    `jax.lax.custom_linear_solve` where the current transpose/adjoint policy
    supports implicit adjoints
  - shell preconditioners can now carry explicit transpose/adjoint callbacks
    through the shared preconditioner policy instead of degrading to forward
    reuse in transpose solves
  - solve/logdet bundles now retain compact transpose-operator metadata in the
    shared `matrix_free_core` auxiliary payload
  - fp64 solve/logdet gradient proof coverage now exists for the implicit
    adjoint path, and a latent-Gaussian Laplace worked example now exists under
    `examples/example_latent_gaussian_laplace.py`

### Core Operator Infrastructure

- `in_progress`
  - direct owner tests now exist for `iterative_solvers`
  - deepen the new `basic` semantics for operator-first surfaces
  - keep extending flexible-preconditioner policy beyond the newly landed
    transpose-aware shell-preconditioner contract
  - keep hardening locking, restart, and correction-equation policy for the
    landed Davidson/Jacobi-Davidson/shift-invert/contour eigensolver tranche,
    especially in the heavier shift-invert and contour eigensolver paths
  - keep extending the new contour-integral/spectral-transform/operator-first
    helper substrate beyond the now-landed `log` / `sqrt` / `root` / `sign`,
    `sin` / `cos` / `sinh` / `cosh` / `tanh`, and `exp` / `tan` wrappers into
    more public `jrb_mat` / `jcb_mat` matrix-function surfaces

### Estimators And Approximation Paths

- `in_progress`
  - evaluate a TFP `fastgp`-style rational multi-shift logdet path for SPD
    operators as a benchmarked alternative to SLQ and Leja+Hutch++
  - evaluate block rational Krylov versus Newton-Leja for GPU-heavy
    matrix-function actions and trace-estimation workloads, with emphasis on
    shift reuse, block RHS structure, stopping criteria, and AD-safe projected
    solves
  - add Hutch++-compatible trace-estimator integration for the rational-Krylov
    path, with variance-reduction and batched-probe support

### AD-Safe Caching And Adjoint Design

- `in_progress`
  - deepen the AD-safe cached rational-Krylov trace/logdet path in the operator
    stack:
    - forward surface should be an operator-plan-first JAX API, not an
      arbitrary callable closure
    - start with `@jax.custom_vjp` rather than a new primitive
    - return compact projected-state metadata only:
      poles, small projected matrices/recurrences, probe metadata,
      deflation/locking maps, and residual estimates
    - a first cached rational Hutch++ metadata layer is now landed; the
      remaining work is the fuller projected-state custom-VJP path rather than
      first-surface creation
    - do not backpropagate through Krylov iteration, restart policy, PSD
      detection, pole selection, or nugget heuristics
    - VJP should use implicit adjoint solves reconstructed from the cached
      projected metadata, with basis reuse across probes where possible
  - make low-rank deflation AD-safe:
    - save the cached deflation state in the forward residual of
      `@jax.custom_vjp` surfaces
    - use the same deflation state in the implicit-adjoint VJP so forward and
      backward approximations stay consistent
    - avoid differentiating through adaptive rank selection, pivot policy, or
      refresh heuristics

### Probe Design And Variance Reduction

- `in_progress`
  - add structured orthogonal probe-block support for stochastic trace/logdet
    estimators across dense, sparse, and matrix-free operator paths:
    - support FWHT/Hadamard-based on-device orthogonal probe blocks as the
      default GPU-friendly variance-reduction path
    - support blocked Rademacher plus thin-QR probe generation as a more
      general fallback when FWHT layout assumptions do not hold
    - make probe-block size a tunable operator-layer hyperparameter, with
      practical power-of-two defaults such as `8`, `16`, `32`, and `64`
    - keep the probe API shared across Hutchinson, Hutch++, SLQ, and
      rational-Krylov trace-estimation paths
    - prefer batched operator apply on whole probe blocks instead of
      per-vector execution
  - add variance-monitoring and adaptive stopping for orthogonal probe blocks:
    - small pilot blocks for empirical variance estimation
    - adaptive probe-budget selection from the pilot variance estimate
    - reproducible per-block PRNG key threading and metadata capture
  - add numerical and implementation rules for orthogonal probe blocks:
    - use normalized orthogonal probes with random signs and optional
      permutations to avoid deterministic alignment with operator structure
    - accumulate stochastic reductions with fp64 or compensated summation
    - allow in-place or streaming FWHT-style generation when memory pressure
      makes full probe-block materialization undesirable
    - keep the probe-generator output JIT-safe and pytree-friendly, with no
      hidden Python-side caching

### Deflation And Recycling

- `in_progress`
  - add low-rank deflation and probe-recycling support for stochastic
    trace/logdet estimation:
    - build and cache compact dominant subspaces using partial Cholesky,
      pivoted Cholesky, Nyström, or related low-rank operator-first methods
    - evaluate the projected low-rank contribution exactly on the compact
      subspace and probe only the residual operator
    - keep extending the newly landed shared deflation metadata beyond current
      residual trace-estimation reuse so Hutchinson, Hutch++, SLQ, and
      rational-Krylov estimators can share the same deflation path
    - recycle the cached subspace across nearby parameter values and repeated
      solves, with refresh rules based on residual mass or variance drift
    - keep deflation metadata on device and pytree-safe rather than relying on
      hidden mutable caches
  - add practical low-rank deflation rules:
    - deflate until the residual carries only a small fraction of the
      trace/log-trace mass or variance proxy
    - re-orthogonalize recycled subspaces periodically when parameter drift
      accumulates
    - prefer matrix-free constructions that require only operator apply and
      optional diagonal estimates

### Numerical Safety And CI

- `in_progress`
  - add numerical safety rules for rational-Krylov logdet/trace work:
    - small Lanczos or Gershgorin pretest for eigen-interval estimation
    - PSD-safe shift and nugget clamping
    - `log1p` handling for tiny projected eigenvalue contributions
    - compensated or pairwise reductions, preferably promoted to fp64 on GPU
  - add CI and contract coverage for cached-adjoint rational-Krylov paths:
    - gradient dot-product checks against finite-difference or complex-step
      references in fp64
    - cached-basis VJP consistency versus explicit implicit-adjoint solves
    - Hutch++ variance scans across probe counts and seeds
    - orthogonal-probe variance scans versus i.i.d. Rademacher probes at fixed
      compute budgets
  - a dedicated lightweight matrix-free tranche test file now exists so the
    newer shared-core contracts can be exercised without routing every change
    through the heaviest chassis modules

## Curvature Layer

Status: `in_progress`

Detailed backlog:
- [curvature_todo.md](/docs/status/curvature_todo.md)

## 6. Special Functions

Status: `in_progress`

Detailed backlog:
- [special_functions_todo.md](/docs/status/special_functions_todo.md)

## Theory And Methodology

Status: `in_progress`

Detailed backlog:
- [theory_todo.md](/docs/status/theory_todo.md)
- [special_functions_todo.md](/docs/status/special_functions_todo.md)

## 7. Analytic / Algebraic / Domain Functionality

Status: `in_progress`

- `done`
  - direct test owners already exist for `acb_dirichlet`, `acb_elliptic`,
    `acb_modular`, `acb_poly`, `arb_poly`, `bernoulli`, `dirichlet`, `dlog`,
    and `partitions`
  - Dirichlet now has dedicated engineering coverage in
    `tests/test_dirichlet_engineering.py`
  - Dirichlet now has a dedicated canonical notebook:
    `examples/example_dirichlet_surface.ipynb`
  - the real Dirichlet rigorous batch path now handles batched tail bounds and
    validity masking correctly in `src/arbplusjax/dirichlet.py`
- `in_progress`
  - continue closing the large missing-C implementation gap in
    `acb_dirichlet`, `arb_poly`, `acb_poly`, `dirichlet`, `acb_modular`, and
    `acb_elliptic`
  - keep domain-family status separate from generic wrapper status so remaining
    work is visible by mathematical family, not just by module plumbing
- `planned`
  - add a clearer per-family gap summary for analytic/algebraic stacks once the
    missing-implementation counts are regenerated

## 8. API / Runtime / Metadata / Validation

Status: `in_progress`

Detailed backlog:
- [api_runtime_todo.md](/docs/status/api_runtime_todo.md)

## Priority test-owner additions

Status: `in_progress`

Detailed backlog:
- [api_runtime_todo.md](/docs/status/api_runtime_todo.md)

## Missing C implementation snapshot

Detailed backlog:
- [api_runtime_todo.md](/docs/status/api_runtime_todo.md)
