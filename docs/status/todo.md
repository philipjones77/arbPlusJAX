Last updated: 2026-03-23T00:00:00Z

# TODO

This is the canonical consolidated TODO list for active implementation work in
`docs/status/`.

Organization rule:
- Sections follow the top-level function and infrastructure categories defined in
  [test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).
- Test-owner priorities follow
  [test_gap_checklist.md](/docs/status/test_gap_checklist.md).
- Generated inventories and detailed provenance stay in `docs/status/reports/`,
  not here.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

Current phase snapshot:
- Tier 0 architecture/API: `in_progress`
- point-fast JAX conversion of public point mode across all six categories: `in_progress`
- Tier 1 core special-function hardening: `in_progress`
- Tier 2 general incomplete-tail engine: `in_progress`
- Tier 3 incomplete Bessel specialization: `in_progress`
- Tier 4 higher extensions: `in_progress`
- multivariate Bessel placement after scalar incomplete infrastructure: `planned`

## Cross-cutting execution and run-platform

Status: `in_progress`

- `done`
  - root-level [tests](/tests) and
    [benchmarks](/benchmarks) remain the
    canonical run surfaces
  - dedicated test orchestration exists in
    [tools/run_test_harness.py](/tools/run_test_harness.py)
  - dedicated benchmark orchestration exists in
    [benchmarks/run_benchmarks.py](/benchmarks/run_benchmarks.py)
    and
    [benchmarks/run_harness_profile.py](/benchmarks/run_harness_profile.py)
  - a shared `runtime_manifest.json` schema now exists across test and benchmark
    outputs
  - the existing
    [example_run_suite.py](/examples/example_run_suite.py)
    flow now writes a suite-level runtime manifest, summary markdown, and SVG
    plots under
    [examples/outputs/example_run_suite/](/examples/outputs/example_run_suite/)
  - canonical notebook execution now exists through
    [run_example_notebooks.py](/tools/run_example_notebooks.py),
    which executes the standards-aligned notebook surfaces and retains
    executed notebooks plus runtime/summary artifacts in the owning
    [examples/outputs/](/examples/outputs/)
    folders
  - canonical notebooks now also encode production calling patterns:
    binder reuse, optional padding/chunking, cached plan reuse, and benchmark
    extension guidance for the main top-level categories
  - the existing official API benchmark
    [benchmark_api_surface.py](/benchmarks/benchmark_api_surface.py)
    now emits the shared benchmark-report JSON schema instead of only printing
    timings
  - Windows, Linux, and Colab run instructions are documented
  - a CPU-safe Colab bootstrap surface now exists in
    [requirements-colab.txt](/requirements-colab.txt)
    and
    [colab_bootstrap.sh](/tools/colab_bootstrap.sh),
    with checked-in platform profile metadata in
    [platform_bootstrap_profiles.json](/configs/platform_bootstrap_profiles.json)
  - bounded CPU validation profiles were re-run through
    [run_test_harness.py](/tools/run_test_harness.py) for `matrix`,
    `special`, and `bench-smoke`; see
    [cpu_validation_profiles.md](/docs/reports/cpu_validation_profiles.md)
  - a dedicated sparse-matrix harness profile exists in
    [run_test_harness.py](/tools/run_test_harness.py) on top of the landed
    sparse point layer
- `in_progress`
  - unify long-run benchmark scheduling and report collection behind a single
    environment manifest and execution policy
  - keep benchmark ownership distinct from correctness ownership, but make the
    pass/fail boundaries clearer in status docs
  - keep docs landing pages, report indexes, status indexes, and current repo
    mapping generated automatically so push/commit does not rely on hand-edited
    tree summaries
  - re-run and retain a bounded sparse CPU validation slice in
    [cpu_validation_profiles.md](/docs/reports/cpu_validation_profiles.md)
  - normalize more legacy benchmark scripts onto the shared benchmark-report
    schema instead of stdout-only summaries; direct normalized coverage now
    includes `benchmark_arb_poly.py`, `benchmark_acb_poly.py`,
    `benchmark_arb_calc.py`, `benchmark_acb_calc.py`,
    `benchmark_dirichlet.py`, and `benchmark_acb_dirichlet.py`
  - keep normalized benchmark CLIs explicitly parameterized for CPU/GPU
    portability and `float32`/`float64` execution, even when the current
    validation slice only runs on CPU
  - continue turning theory coverage into first-class tranche status rather than
    leaving methodology gaps implicit
- `planned`
  - add a single repo-facing execution checklist that names the minimum CPU,
    parity, GPU, and benchmark slices required for a release-quality change

## Cross-cutting point-fast JAX conversion

Status: `in_progress`

- `done`
  - the repo now has an explicit definition of `fast JAX` for point mode in
    [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
  - the six-category implementation program now exists in
    [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)
  - the six-category status plan now exists in
    [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)
  - the required six-category coverage matrix now exists in
    [point_fast_jax_category_matrix.md](/docs/reports/point_fast_jax_category_matrix.md)
  - all public point functions now have compiled single-call, compiled batch,
    and family-owned direct batch public surfaces; see
    [point_fast_jax_function_inventory.md](/docs/reports/point_fast_jax_function_inventory.md)
  - family-level direct-batch proof coverage now exists for the previously
    generic-batch incomplete-tail set:
    `incomplete_gamma_upper`, `incomplete_gamma_lower`,
    `incomplete_bessel_i`, `incomplete_bessel_k`, and
    `laplace_bessel_k_tail`
- `in_progress`
  - create the per-category point-mode audit that classifies current surfaces as
    `direct_fast`, `recurrence_fast`, `approx_fast`, or `precise_only_for_now`
  - widen the now-landed category and incomplete-tail proof slices into deeper
    family-by-family numerical proof coverage across the remaining large public
    matrix/core/hypergeometric surfaces
  - refactor point kernels so Python control flow, dynamic shapes, Arb objects,
    and precise fallback logic remain outside the hot path
  - build shared point-fast infrastructure for logspace, recurrence,
    approximants, and region routing
- `planned`
  - add machine-readable point-fast capability metadata for downstream routing

## Cross-repo provider boundary

Status: `in_progress`

- `done`
  - arbPlusJAX remains the hardened numeric-kernel repo rather than being
    repurposed as another library's orchestration layer
  - matrix, sparse, block/vblock, and matrix-free/operator infrastructure are
    being hardened as repo-owned numeric infrastructure inside arbPlusJAX
- `in_progress`
  - keep hardening public provider-worthy families instead of exposing more ad
    hoc module-internal integration paths
  - prefer stable capability entrypoints and metadata-bearing public surfaces
    over downstream imports of internal module layout
  - strengthen metadata and diagnostics so downstream orchestration can route
    intelligently on method, hardening level, derivative support, and runtime
    strategy
  - keep notebooks, tests, and benchmarks written as downstream-consumer
    documentation and validation surfaces, not only as internal development
    checks
  - keep cross-repo integration thin: downstream libraries should integrate
    through adapter/provider layers on their side rather than by restructuring
    arbPlusJAX around a specific consumer
  - Barnes/double-gamma now has explicit downstream capability aliases through
    the IFJ-compatible public surface; continue tightening diagnostics and
    narrower provider wording around that capability
  - fragile-regime promotion hooks now have explicit downstream capability
    aliases for incomplete gamma upper and incomplete Bessel `I`/`K`;
    continue narrowing terminology and diagnostics expectations around those
    hooks
  - make incomplete-Bessel provider-grade next
- `planned`
  - document a narrower capability-contract surface specifically for
    downstream-provider use once the Barnes/promotion/incomplete-Bessel tranche
    is hardened enough to freeze terminology

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
- `planned`
  - add a repo-wide wrapper contract matrix covering shape normalization,
    broadcasting, dtype promotion, batch padding, and mode dispatch invariants

## 3. Dense Matrix Functionality

Status: `in_progress`

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
    [dense_matrix_surface_benchmark.md](/docs/status/reports/dense_matrix_surface_benchmark.md),
    and
    [example_dense_matrix_surface.ipynb](/examples/example_dense_matrix_surface.ipynb)
- `in_progress`
  - strengthen large-`n` determinant enclosures beyond midpoint fallback
  - exact-reference dense chassis checks now cover `inv`, `qr`, cached matvec,
    determinant, trace, and matrix norms for real and complex dense matrices
  - dense chassis coverage now also includes point/basic/JIT/API parity checks
    for banded matvec on real and complex paths
  - keep expanding larger-`n` determinant enclosure quality beyond midpoint
    fallback and current 5x5/6x6 reference-nesting checks
  - separate midpoint-first solve-family implementations from true interval/box
    linear-algebra kernels in status and engineering reports
  - deepen rigorous treatment for solve/inverse/factorization paths instead of
    relying on wrapper-level outward boxing around midpoint kernels
- `planned`
  - align dense matrix status reporting with the same engineering fields used by
    function status reports: hardening, AD audit, batch support, and pure-JAX
    aspiration

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
- `in_progress`
  - direct owner tests now exist for `sparse_common`
- `planned`
  - add interval/box block-sparse and variable-block sparse modes
  - add more benchmark coverage that separates storage-format overhead from
    solver quality

## 5. Matrix-Free / Operator Functionality

Status: `in_progress`

### Current Landed Surface

- `done`
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

### Core Operator Infrastructure

- `in_progress`
  - direct owner tests now exist for `iterative_solvers`
  - deepen the new `basic` semantics for operator-first surfaces
  - harden flexible-preconditioner policy beyond the current shared
    preconditioned `minres` path
  - harden locking, restart, and correction-equation policy for the landed
    Davidson/Jacobi-Davidson/shift-invert/contour eigensolver tranche
  - keep extending the new contour-integral/spectral-transform/operator-first
    helper substrate from the current core action/combiner layer into more
    public `jrb_mat` / `jcb_mat` matrix-function surfaces

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
  - add an AD-safe cached rational-Krylov trace/logdet path in the operator
    stack:
    - forward surface should be an operator-plan-first JAX API, not an
      arbitrary callable closure
    - start with `@jax.custom_vjp` rather than a new primitive
    - return compact projected-state metadata only:
      poles, small projected matrices/recurrences, probe metadata,
      deflation/locking maps, and residual estimates
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
    - expose a reusable deflated-operator plan so Hutchinson, Hutch++, SLQ, and
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
    - FWHT and QR probe-block parity checks for unbiased trace estimation on
      small reference problems
    - low-rank-deflated estimator variance scans versus undeflated estimators
      at fixed probe budgets
    - cached-deflation forward/VJP consistency checks across nearby parameter
      values
    - eigen-interval and nugget sensitivity sweeps for logdet stability
    - JIT/cache stability checks so value changes do not trigger recompiles for
      fixed operator shapes/configuration
    - pytree contract checks so cached aux metadata survives `jit`, `vmap`, and
      repeated calls without hidden Python-side state

### Longer-Horizon Additions

- `planned`
  - add contour-integral matrix functions plus reusable dense/operator-first
    `logm`, `sqrtm`, `rootm`, and `signm` infrastructure
  - add broader operator-parameter adjoints beyond the now-landed
    parameter-differentiable operator-plan tranche
  - keep PETSc/SLEPc as benchmark and design references only, not governed
    runtime backends

## 6. Special Functions

Status: `in_progress`

- `done`
  - canonical example notebooks now exist for top-level gamma and
    Barnes/double-gamma production surfaces, with explicit production-calling
    guidance and benchmark-extension notes
  - dedicated theory notes now exist for the gamma-family production stack
  - dedicated theory notes now exist for hypergeometric and Barnes/double-gamma
    production methodology
- `in_progress`
  - normalize special-function service benchmarks and diagnostics reporting more
    fully across hypergeometric, Bessel, gamma, and Barnes families
  - direct normalized special benchmark coverage now also includes
    `benchmark_hypgeom_extra.py`
  - continue converting notebook and benchmark guidance into schema-backed
    artifacts rather than stdout-only summaries

## Theory And Methodology

Status: `in_progress`

- `done`
  - theory index now reflects production-readiness interpretation rather than
    only listing older notes
  - dedicated methodology notes now exist for gamma-family and transform/NUFFT
    surfaces
  - dedicated methodology notes now exist for sparse/block/vblock and
    matrix-free production surfaces

- `done`
  - canonical public entry points exist for `tail_integral`,
    `tail_integral_accelerated`, `incomplete_bessel_k`,
    `incomplete_bessel_i`, `incomplete_gamma_upper`,
    `incomplete_gamma_lower`, and `laplace_bessel_k_tail`
  - generic tail-engine modules exist under
    `src/arbplusjax/special/tail_acceleration/`
  - incomplete-Bessel modules exist under `src/arbplusjax/special/bessel/`
  - pure-JAX `high_precision_refine` is exposed, with `mpfallback` retained
    only as a compatibility alias
  - explicit derivative support exists for incomplete gamma, Laplace-Bessel
    tails, and the current incomplete `K` / incomplete `I` point paths
  - `boost_hypgeom` and `cusf_compat` surfaces now exist with tests and docs
- `in_progress`
  - continue hardening ordinary gamma, Barnes-family, and ordinary Bessel
    families where coverage remains uneven
  - continue calibrating the generic tail-engine recurrence and sequence logic
    across more families
  - bring incomplete `I` to the same regime maturity as incomplete `K`
  - finish hardening and characterization of the `bdg_*` Barnes and
    double-gamma family
  - reduce runtime cost of rigorous/adaptive `bdg_*` samplers in
    `src/arbplusjax/ball_wrappers.py`
  - continue hypergeometric engineering cleanup:
    helper consolidation, family-specific adaptive/rigorous kernels, and
    compile-noise reduction outside the current representative families
  - `pfq` fixed/padded basic and adaptive/rigorous mode-batch proofs are now
    landed on the canonical real/complex paths
  - alternative hypergeometric hardening is now stronger:
    Boost `pfq` fixed/padded mode-batch proofs and helper/`pfq` point-AD smoke
    are landed, and CuSF `hyp1f1`/`hyp2f1` now have explicit mode containment
    plus point-AD checks
  - regularized Boost `0f1`/`1f1` fixed-vs-padded containment and reciprocal
    `pfq` fixed-vs-padded/mode-containment proofs are now landed
  - Boost helper aliases now have explicit cross-mode consistency checks
  - direct owner tests now exist for `bessel_kernels` and `barnesg`
  - extend benchmark and RF77-facing usage/report coverage where diagnostics
    exist but packaging is still incomplete
- `planned`
  - extend the general incomplete-tail engine to more hypergeometric-tail
    families
  - add multivariate Bessel work only after scalar incomplete infrastructure is
    genuinely stable
  - add incomplete multivariate-Bessel-type routines only if reductions justify
    them
  - resolve what true arbitrary precision means under a strict pure-JAX
    constraint

Priority rule for remaining Arb/FLINT-style breadth:
- do not migrate the missing callable surface breadth-first
- prefer one canonical JAX-native implementation per important function family
- prioritize IFJ and RF77-facing work first:
  - Barnes-family hardening and IFJ-derived double-gamma work
  - gamma-adjacent continuation functions that unblock contour and residue
    workflows
  - selected complex special functions with direct downstream use:
    `Ei`, `Chi`, `Ci`, dilogarithm, Tricomi `U`, and selected `pfq`
- after that, prioritize broad-value parity:
  - dense matrix parity in `arb_mat` / `acb_mat`
  - polynomial parity in `arb_poly` / `acb_poly`
  - selected scalar gaps such as `lambertw`, zeta-adjacent functions, and
    rising/beta families
- defer broad elliptic/modular and full Dirichlet/L-function expansion until
  the Barnes/gamma/integration path is stable enough to justify it

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

- `done`
  - public metadata exists for family, stability, method tags, regime tags, and
    derivative status
  - direct test owners already exist for `api`, `runtime`, `elementary`,
    `function_provenance`, `jax_diagnostics`, `jax_precision`, `cusf_compat`,
    and `soft_ops`
  - direct owner tests now exist for `public_metadata` and
    `capability_registry`
  - direct owner tests now exist for `checks`, `coeffs`, `precision`,
    `validation`, and `soft_types`
  - public metadata now supports explicit filtering and deterministic JSON
    serialization for report-facing and downstream-adapter usage
- `in_progress`
  - keep stable versus experimental API status explicit in metadata and status
    reports
  - keep naming cleanup moving toward canonical mathematical names with
    implementation selection, rather than provenance-prefixed public names
  - keep `docs/implementation/modules/arb_mat_implementation.md`,
    `docs/implementation/modules/acb_mat_implementation.md`,
    `docs/implementation/modules/jrb_mat_implementation.md`, and
    `docs/implementation/modules/jcb_mat_implementation.md` aligned with source

## Priority test-owner additions

Status: `in_progress`

Highest priority:
- landed
  [test_ball_wrappers_contracts.py](/tests/test_ball_wrappers_contracts.py)
- landed
  [test_baseline_wrappers_contracts.py](/tests/test_baseline_wrappers_contracts.py)
- landed
  [test_mat_wrappers_contracts.py](/tests/test_mat_wrappers_contracts.py)
- landed
  [test_wrappers_common_contracts.py](/tests/test_wrappers_common_contracts.py)
- landed
  [test_double_interval_wrappers_contracts.py](/tests/test_double_interval_wrappers_contracts.py)
- landed
  [test_core_wrappers_contracts.py](/tests/test_core_wrappers_contracts.py)
- landed
  [test_calc_wrappers_contracts.py](/tests/test_calc_wrappers_contracts.py)
- landed
  [test_checks_contracts.py](/tests/test_checks_contracts.py)
- landed
  [test_coeffs_contracts.py](/tests/test_coeffs_contracts.py)
- landed
  [test_precision_contracts.py](/tests/test_precision_contracts.py)
- landed
  [test_kernel_helpers_contracts.py](/tests/test_kernel_helpers_contracts.py)
- landed
  [test_mat_common_contracts.py](/tests/test_mat_common_contracts.py)
- landed
  [test_sampling_helpers_contracts.py](/tests/test_sampling_helpers_contracts.py)
- landed
  [test_sparse_common_contracts.py](/tests/test_sparse_common_contracts.py)
- landed
  [test_iterative_solvers_contracts.py](/tests/test_iterative_solvers_contracts.py)
- landed
  [test_soft_types_contracts.py](/tests/test_soft_types_contracts.py)
- landed
  [test_validation_contracts.py](/tests/test_validation_contracts.py)

Next priority:
- landed
  [test_krylov_solvers_contracts.py](/tests/test_krylov_solvers_contracts.py)
- landed
  [test_transform_common_contracts.py](/tests/test_transform_common_contracts.py)
- landed
  [test_dft_engineering.py](/tests/test_dft_engineering.py)
  for cached DFT/NUFFT plan reuse, padded-batch stability, wrapper-mode
  containment, diagnostics, and point/basic AD smoke

Execution order:
1. Re-run the CPU chassis and profile suite after the newly landed direct-owner
   tests.
2. Expand AD and compile-behavior assertions where those tests expose weak
   spots.
3. Identify the next shared-infrastructure tranche that still lacks a direct
   owner.
4. Add that next tranche of focused tests.
5. Re-run the full CPU harness, then opt-in parity and benchmark slices.

## Missing C implementation snapshot

Source:
[audit.md](/docs/status/audit.md)
snapshot `2026-02-25T03:51:38Z`.

- Arb Core: 195
- ACB Core: 144
- ARF: 95
- MAG: 78
- ACB Mat: 110
- Arb Mat: 109
- FMPR: 59
- ACB Dirichlet: 87
- Arb Poly: 87
- ACB Poly: 86
- Dirichlet: 38
- Bool Mat: 34
- ACB Modular: 27
- ACB Elliptic: 17
- ACF: 10
- ACB DFT: 0
- Arb Calc: 0
- ACB Calc: 0

Use the snapshot above for gap sizing only. Detailed missing-symbol inventories
belong in `docs/status/reports/missing_impls/`.
