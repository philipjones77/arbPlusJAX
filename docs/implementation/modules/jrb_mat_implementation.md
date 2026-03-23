Last updated: 2026-03-22T00:00:00Z

# jrb_mat

## Role

`jrb_mat` is the Jones-labeled subsystem for real matrix-free JAX algorithms.

It is separate from [arb_mat](/src/arbplusjax/arb_mat.py):
- `arb_mat`: canonical Arb/FLINT-style JAX matrix extension surface
- `jrb_mat`: operator-style matrix-free subsystem for Krylov actions, trace estimators, and large-scale AD-aware workflows

## Layout Contracts

Canonical layouts:
- matrix: `(..., n, n, 2)`
- vector: `(..., n, 2)`

Interpretation:
- trailing `2` stores real intervals as `[lo, hi]`
- matrices must be square
- vectors are column-like logical vectors stored as rank-1 interval arrays

Public contract helpers:
- `jrb_mat_as_interval_matrix(a)`
- `jrb_mat_as_interval_vector(x)`
- `jrb_mat_shape(a)`

## Current Implemented Substrate

Point mode:
- `jrb_mat_matmul_point(a, b)`
- `jrb_mat_matvec_point(a, x)`
- `jrb_mat_solve_point(a, b)`
- `jrb_mat_triangular_solve_point(a, b, lower=...)`
- `jrb_mat_lu_point(a)`

Basic mode:
- `jrb_mat_matmul_basic(a, b)`
- `jrb_mat_matvec_basic(a, x)`
- `jrb_mat_solve_basic(a, b)`
- `jrb_mat_triangular_solve_basic(a, b, lower=...)`
- `jrb_mat_lu_basic(a)`

Precision/JIT entry points:
- `jrb_mat_matmul_basic_prec`
- `jrb_mat_matvec_basic_prec`
- `jrb_mat_solve_basic_prec`
- `jrb_mat_triangular_solve_basic_prec`
- `jrb_mat_lu_basic_prec`
- `jrb_mat_matmul_basic_jit`
- `jrb_mat_matvec_basic_jit`
- `jrb_mat_solve_basic_jit`
- `jrb_mat_triangular_solve_basic_jit`
- `jrb_mat_lu_basic_jit`

Matrix-free operator and Krylov layer:
- `jrb_mat_dense_operator(a)`
- `jrb_mat_dense_operator_adjoint(a)`
- `jrb_mat_bcoo_operator(a)`
- `jrb_mat_bcoo_operator_adjoint(a)`
- `jrb_mat_bcoo_parametric_operator(indices, shape=...)`
- `jrb_mat_dense_operator_plan_prepare(a)`
- `jrb_mat_dense_operator_rmatvec_plan_prepare(a)`
- `jrb_mat_dense_operator_adjoint_plan_prepare(a)`
- `jrb_mat_shell_operator_plan_prepare(callback, context=...)`
- `jrb_mat_finite_difference_operator_plan_prepare(function, base_point=..., ...)`
- `jrb_mat_finite_difference_operator_plan_set_base(plan, base_point=..., ...)`
- `jrb_mat_sparse_operator_plan_prepare(a)`
- `jrb_mat_sparse_operator_rmatvec_plan_prepare(a)`
- `jrb_mat_sparse_operator_adjoint_plan_prepare(a)`
- `jrb_mat_block_sparse_operator_plan_prepare(a)`
- `jrb_mat_vblock_sparse_operator_plan_prepare(a)`
- `jrb_mat_bcoo_parametric_operator_plan_prepare(indices, data, shape=...)`
- `jrb_mat_scipy_csr_operator(csr)`
- `jrb_mat_bcoo_gershgorin_bounds(a, eps=...)`
- `jrb_mat_bcoo_spectral_bounds_adaptive(a, steps=..., safety_margin=...)`
- `jrb_mat_operator_apply_point(matvec, x)`
- `jrb_mat_operator_apply_basic(matvec, x)`
- `jrb_mat_poly_action_point(matvec, x, coefficients)`
- `jrb_mat_expm_action_point(matvec, x, terms=...)`
- `jrb_mat_expm_action_lanczos_restarted_point(matvec, x, steps=..., restarts=...)`
- `jrb_mat_expm_action_lanczos_block_point(matvec, xs, steps=..., restarts=...)`
- `jrb_mat_expm_action_lanczos_restarted_with_diagnostics_point(matvec, x, steps=..., restarts=...)`
- `jrb_mat_lanczos_tridiag_point(matvec, x, steps)`
- `jrb_mat_lanczos_diagnostics_point(matvec, x, steps)`
- `jrb_mat_funm_action_lanczos_point(matvec, x, dense_funm, steps)`
- `jrb_mat_funm_action_lanczos_with_diagnostics_point(matvec, x, dense_funm, steps)`
- `jrb_mat_funm_integrand_lanczos_point(matvec, x, dense_funm, steps)`
- `jrb_mat_trace_integrand_point(matvec, x)`
- `jrb_mat_funm_trace_integrand_lanczos_point(matvec, x, scalar_fun, steps)`
- `jrb_mat_trace_estimator_point(matvec, probes)`
- `jrb_mat_trace_estimator_with_diagnostics_point(matvec, probes)`
- `jrb_mat_logdet_slq_point(matvec, probes, steps)`
- `jrb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps)`
- `jrb_mat_logdet_solve_point(matvec, rhs, ...)`
- `jrb_mat_logdet_solve_basic(matvec, rhs, ...)`
- `jrb_mat_log_action_leja_point(matvec, x, degree=..., spectral_bounds=...)`
- `jrb_mat_log_action_leja_with_diagnostics_point(...)`
- `jrb_mat_hutchpp_trace_point(action_fn, sketch_probes, residual_probes)`
- `jrb_mat_logdet_leja_hutchpp_point(matvec, sketch_probes, residual_probes, degree=..., spectral_bounds=...)`
- `jrb_mat_logdet_leja_hutchpp_with_diagnostics_point(...)`
- `jrb_mat_bcoo_logdet_leja_hutchpp_point(a, sketch_probes, residual_probes, ...)`
- `jrb_mat_bcoo_logdet_leja_hutchpp_with_diagnostics_point(a, sketch_probes, residual_probes, ...)`
- `jrb_mat_bcoo_inverse_diagonal_point(a, overlap=..., block_size=..., correction_probes=...)`
- `jrb_mat_bcoo_inverse_diagonal_with_diagnostics_point(...)`
- `jrb_mat_minres_solve_action_point(matvec, rhs, preconditioner=..., **kwargs)`
- `jrb_mat_minres_inverse_action_point(matvec, rhs, preconditioner=..., **kwargs)`
- `jrb_mat_multi_shift_solve_point(matvec, rhs, shifts, symmetric=..., **kwargs)`
- `jrb_mat_multi_shift_solve_symmetric_point(matvec, rhs, shifts, **kwargs)`
- `jrb_mat_multi_shift_solve_spd_point(matvec, rhs, shifts, **kwargs)`
- `jrb_mat_multi_shift_solve_basic(matvec, rhs, shifts, symmetric=..., **kwargs)`
- `jrb_mat_eigsh_point(matvec, size=..., k=..., which=..., steps=...)`
- `jrb_mat_eigsh_with_diagnostics_point(matvec, size=..., k=..., which=..., steps=...)`
- `jrb_mat_eigsh_restarted_point(matvec, size=..., k=..., which=..., steps=..., restarts=..., block_size=...)`
- `jrb_mat_eigsh_restarted_with_diagnostics_point(...)`
- `jrb_mat_eigsh_block_point(matvec, size=..., k=..., which=..., steps=..., block_size=...)`
- `jrb_mat_eigsh_block_with_diagnostics_point(...)`
- `jrb_mat_eigsh_krylov_schur_point(matvec, size=..., k=..., which=..., steps=..., restarts=..., block_size=...)`
- `jrb_mat_eigsh_krylov_schur_with_diagnostics_point(...)`
- `jrb_mat_eigsh_davidson_point(matvec, size=..., k=..., which=..., subspace_iters=..., block_size=..., preconditioner=...)`
- `jrb_mat_eigsh_davidson_with_diagnostics_point(...)`
- `jrb_mat_eigsh_jacobi_davidson_point(matvec, size=..., k=..., which=..., subspace_iters=..., block_size=..., preconditioner=...)`
- `jrb_mat_eigsh_jacobi_davidson_with_diagnostics_point(...)`
- `jrb_mat_generalized_operator_plan_prepare(a_matvec, b_matvec, b_preconditioner=..., tol=..., atol=..., maxiter=...)`
- `jrb_mat_geigsh_point(a_matvec, b_matvec, size=..., k=..., which=..., steps=..., b_preconditioner=...)`
- `jrb_mat_geigsh_with_diagnostics_point(...)`
- `jrb_mat_eigsh_shift_invert_point(matvec, size=..., shift=..., k=..., which=..., steps=..., preconditioner=...)`
- `jrb_mat_eigsh_shift_invert_with_diagnostics_point(...)`
- `jrb_mat_generalized_shift_invert_operator_plan_prepare(a_matvec, b_matvec, shift=..., preconditioner=..., tol=..., atol=..., maxiter=...)`
- `jrb_mat_geigsh_shift_invert_point(a_matvec, b_matvec, size=..., shift=..., k=..., which=..., steps=..., preconditioner=...)`
- `jrb_mat_geigsh_shift_invert_with_diagnostics_point(...)`
- `jrb_mat_neigsh_point(matvec_builder, dmatvec_builder, size=..., lambda0=..., newton_iters=..., eig_steps=...)`
- `jrb_mat_neigsh_with_diagnostics_point(...)`
- `jrb_mat_peigsh_point(coeff_matvecs, size=..., lambda0=..., newton_iters=..., eig_steps=...)`
- `jrb_mat_peigsh_with_diagnostics_point(...)`
- `jrb_mat_eigsh_contour_point(matvec, size=..., center=..., radius=..., k=..., which=..., quadrature_order=..., block_size=..., preconditioner=...)`
- `jrb_mat_eigsh_contour_with_diagnostics_point(...)`
- `jrb_mat_rademacher_probes_like(x, key=..., num=...)`
- `jrb_mat_normal_probes_like(x, key=..., num=...)`

## Current Methodology

Point:
- midpoint real linear algebra
- result boxed back to an outward interval

Basic:
- `matmul` / `matvec` use interval arithmetic on the canonical `(..., 2)` layout
- `solve_basic` currently uses midpoint solve plus outward boxing
- `triangular_solve_basic` currently uses midpoint triangular solve plus outward boxing
- `lu_basic` currently uses midpoint LU plus outward boxing of `P`, `L`, `U`

This means:
- `matmul_basic` and `matvec_basic` are genuine interval substrate operations
- `solve_basic` is currently a first substrate implementation, not a final rigorous interval linear solve

Matrix-free layer:
- Lanczos is implemented in pure JAX with fixed-shape scan state and explicit full-basis reorthogonalisation.
- `funm_action` and quadratic-form pathways are action-first. They target `f(A)b`, trace estimators, and SLQ-style `logdet`, not dense `logm(A)`.
- sparse SPD operators can now be wired through JAX `BCOO` closures directly, and fixed-pattern gather/segment-sum operator closures exist for later parameter-differentiable sparse work
- the adjacent sparse point layer (`srb_mat`) now supports callable left preconditioners on the `cg` path, which is the immediate hook for Jacobi or other simple JAX-native preconditioners in outer RF77 workflows
- sparse SPD logdet now also has a Leja-interpolation route: `log(A)v` is approximated by Newton interpolation on a transformed Leja interval, and the trace is estimated with Hutch++
- that Leja route now supports adaptive truncation through `max_degree`, `min_degree`, `rtol`, and `atol`
- the Leja setup is now centralized in shared helpers: spectral scaling, transformed-node coefficient generation, and Newton action evaluation all reuse the same path
- coordinate-aligned probe vectors now take an exact shortcut when the sparse action reveals a positive scalar eigenpair, which stabilizes diagonal and basis-aligned sparse trace/logdet cases without pretending they consumed a full adaptive Leja budget
- for sparse `BCOO` inputs, conservative Gershgorin spectral bounds are available directly from the sparse structure, and a multi-start short-Lanczos heuristic can narrow the interpolation interval in point mode
- a sparse `BCOO` convenience wrapper now combines operator construction, automatic bound selection, adaptive Leja action evaluation, and Hutch++ trace estimation in one entry point
- sparse inverse-diagonal estimation now has a first JAX-native selected-inversion-style entry point: contiguous seed blocks are expanded by overlap hops, local inverse rows are formed on the overlap blocks, and an optional stochastic correction estimates the residual diagonal through full-operator solves
- the local selected-inverse rows are also reused as a JAX-native left preconditioner for the correction solves
- this current selected-inversion subset is point-mode only and currently targets `diag(A^{-1})`; selected off-diagonal entries and `tr(A^{-1} dA)`-style surfaces remain future work
- restarted Krylov support now exists for the matrix-free `expm` action path via repeated scaled-action application
- block right-hand-side support now exists for the restarted `expm` action path through a batched matrix-free wrapper
- multi-shift solve support now exists as a shared operator-plan path and is the current substrate for shifted solve benchmarking and future rational matrix-function work
- partial-spectrum eigensolver support now includes restarted, block, Davidson, Jacobi-Davidson, shift-invert, and contour-filter entry points on the same operator substrate, but these families are still early implementations rather than fully hardened solver products
- generalized Hermitian-definite partial-spectrum support now exists through `jrb_mat_geigsh_*`, including a generalized shift-invert spectral-transform path that applies `(A - sigma B)^{-1} B`
- polynomial and nonlinear Hermitian point surfaces now exist through Newton refinement on the smallest-magnitude shift-invert eigenpair, with polynomial operators built automatically from coefficient operator families
- those eigensolver paths now also expose diagnostics surfaces and the current Davidson/Jacobi-Davidson block retains and expands the trial subspace instead of collapsing back to a fixed-width basis each iteration
- `logdet` and solve can now be returned together through `jrb_mat_logdet_solve_*`, with shared auxiliary metadata packaged in `matrix_free_core`
- operator plans now also cover shell callbacks, finite-difference Jacobian-vector products, block/vblock sparse adapters, and a parameter-differentiable sparse `BCOO` closure path
- Backward support now exists for the input/probe vector pathways through custom VJPs on:
  - `jrb_mat_funm_action_lanczos_point`
  - `jrb_mat_funm_integrand_lanczos_point`
  - `jrb_mat_trace_integrand_point`
- Estimator gradients flow through those custom VJPs, so `trace_estimator_point` and `logdet_slq_point` are probe-differentiable without tracing naively through every Krylov loop state.

## Diagnostics And Benchmarks

Current correctness coverage:
- [test_jrb_mat_chassis.py](/tests/test_jrb_mat_chassis.py)
- includes operator apply, polynomial action, `expm` action, Lanczos exact diagonal cases, trace/logdet estimators, and backward probe-gradient checks
- [test_jrb_mat_logdet_contracts.py](/tests/test_jrb_mat_logdet_contracts.py)
- adds SLQ/logdet sanity contracts for diagonal exactness, eigen-tail sensitivity, probe-budget variance, reproducibility/dtype stability, sparse Leja+Hutch++ diagonal exactness, and the new auto-bounds adaptive-degree sparse diagonal contract
- [test_jrb_mat_selected_inverse.py](/tests/test_jrb_mat_selected_inverse.py)
- adds selected-inverse contracts for sparse diagonal exactness, full-overlap inverse-diagonal exactness, and stochastic-correction improvement over the local-only estimate

Current benchmark coverage:
- [benchmark_matrix_free_krylov.py](/benchmarks/benchmark_matrix_free_krylov.py)
- current report: [matrix_free_krylov_benchmark.md](/docs/status/reports/matrix_free_krylov_benchmark.md)
- optional JSON sanity snapshot: `python tools/slq_logdet_contract_report.py`
- broader method comparison note: [matrix_logdet_landscape_implementation.md](/docs/implementation/matrix_logdet_landscape_implementation.md)
- theory note for the sparse SPD Leja path: [sparse_symmetric_leja_hutchpp_logdet.md](/docs/theory/sparse_symmetric_leja_hutchpp_logdet.md)

Current diagnostic contract:
- structured diagnostics now exist via `JrbMatKrylovDiagnostics`
- current fields include:
  - algorithm code
  - steps
  - basis dimension
  - restart count
  - initial norm `beta0`
  - tail norm
  - breakdown flag
  - adjoint-usage flag
  - gradient-support flag
- probe count
  - primal residual
  - adjoint residual
  - regime/method/solver/structure codes
  - convergence flag
  - locked-count summary
  - convergence metric
- for Leja plus Hutch++ runs, `steps` records the used polynomial degree on the representative probe and `tail_norm` records the last Newton increment norm
- `algorithm_code = 3` denotes the ordinary Leja plus Hutch++ path
- `algorithm_code = 4` denotes the exact coordinate/eigenvector shortcut on the representative probe; in that case `steps = 1`
- the current diagnostic surface also includes:
  - explicit requirement that operator callbacks are pure-JAX and fixed-shape
  - explicit distinction between forward action timing and backward gradient timing in the benchmark runner
  - explicit `adjoint` separation at the operator level on the complex side, while the real symmetric side reuses the same operator
  - verified hot-path implementation remains inside JAX execution: no SciPy calls, no NumPy host kernels, no callback ops, and no Python loops in the Krylov core

## Not Yet Implemented / Not Yet Hardened

Planned matrix-function families:
- `jrb_mat_logm`
- `jrb_mat_sqrtm`
- `jrb_mat_rootm`
- `jrb_mat_signm`

Planned lower-level substrate:
- `qr`
- Hessenberg / Schur-compatible reductions

Still missing or incomplete on the current public matrix-free path:
- broader operator-parameter adjoints beyond the sparse parametric operator-plan path
- fully specified `basic` enclosure policy for stochastic estimators and solve-action families
- mature locking / deflation / convergence-history policy for the newer eigensolver tranche
- recycled Krylov basis reuse for repeated adjoints and rational matrix-function workflows

## Design Intent

- obey repo dtype, batching, and AD rules
- keep matrix substrate separate from the canonical Arb-like `arb_mat` namespace
- make the substrate reusable for matrix-free Krylov, stochastic trace estimation, and later RF77-facing large-scale operator workflows
