Last updated: 2026-03-22T00:00:00Z

# jcb_mat

## Role

`jcb_mat` is the Jones-labeled subsystem for complex matrix-free JAX algorithms.

It is separate from [acb_mat](/src/arbplusjax/acb_mat.py):
- `acb_mat`: canonical Arb/FLINT-style JAX extension surface for complex box matrices
- `jcb_mat`: operator-style matrix-free subsystem for Arnoldi actions, trace estimators, and AD-aware large-scale complex workflows

## Layout Contracts

Canonical layouts:
- matrix: `(..., n, n, 4)`
- vector: `(..., n, 4)`

Interpretation:
- trailing `4` stores complex boxes as `[re_lo, re_hi, im_lo, im_hi]`
- matrices must be square
- vectors are logical column vectors stored in box layout

Public contract helpers:
- `jcb_mat_as_box_matrix(a)`
- `jcb_mat_as_box_vector(x)`
- `jcb_mat_shape(a)`

## Current Implemented Substrate

Point mode:
- `jcb_mat_matmul_point(a, b)`
- `jcb_mat_matvec_point(a, x)`
- `jcb_mat_solve_point(a, b)`
- `jcb_mat_triangular_solve_point(a, b, lower=...)`
- `jcb_mat_lu_point(a)`

Basic mode:
- `jcb_mat_matmul_basic(a, b)`
- `jcb_mat_matvec_basic(a, x)`
- `jcb_mat_solve_basic(a, b)`
- `jcb_mat_triangular_solve_basic(a, b, lower=...)`
- `jcb_mat_lu_basic(a)`

Precision/JIT entry points:
- `jcb_mat_matmul_basic_prec`
- `jcb_mat_matvec_basic_prec`
- `jcb_mat_solve_basic_prec`
- `jcb_mat_triangular_solve_basic_prec`
- `jcb_mat_lu_basic_prec`
- `jcb_mat_matmul_basic_jit`
- `jcb_mat_matvec_basic_jit`
- `jcb_mat_solve_basic_jit`
- `jcb_mat_triangular_solve_basic_jit`
- `jcb_mat_lu_basic_jit`

Matrix-free operator and Krylov layer:
- `jcb_mat_dense_operator(a)`
- `jcb_mat_dense_operator_adjoint(a)`
- `jcb_mat_dense_operator_plan_prepare(a)`
- `jcb_mat_dense_operator_rmatvec_plan_prepare(a)`
- `jcb_mat_dense_operator_adjoint_plan_prepare(a)`
- `jcb_mat_shell_operator_plan_prepare(callback, context=...)`
- `jcb_mat_finite_difference_operator_plan_prepare(function, base_point=..., ...)`
- `jcb_mat_finite_difference_operator_plan_set_base(plan, base_point=..., ...)`
- `jcb_mat_sparse_operator_plan_prepare(a)`
- `jcb_mat_sparse_operator_rmatvec_plan_prepare(a)`
- `jcb_mat_sparse_operator_adjoint_plan_prepare(a)`
- `jcb_mat_block_sparse_operator_plan_prepare(a)`
- `jcb_mat_vblock_sparse_operator_plan_prepare(a)`
- `jcb_mat_bcoo_parametric_operator_plan_prepare(indices, data, shape=...)`
- `jcb_mat_operator_apply_point(matvec, x)`
- `jcb_mat_operator_apply_basic(matvec, x)`
- `jcb_mat_poly_action_point(matvec, x, coefficients)`
- `jcb_mat_expm_action_point(matvec, x, terms=...)`
- `jcb_mat_expm_action_arnoldi_restarted_point(matvec, x, steps=..., restarts=..., adjoint_matvec=...)`
- `jcb_mat_expm_action_arnoldi_block_point(matvec, xs, steps=..., restarts=..., adjoint_matvec=...)`
- `jcb_mat_expm_action_arnoldi_restarted_with_diagnostics_point(matvec, x, steps=..., restarts=..., adjoint_matvec=...)`
- `jcb_mat_arnoldi_hessenberg_point(matvec, x, steps)`
- `jcb_mat_arnoldi_diagnostics_point(matvec, x, steps, used_adjoint=...)`
- `jcb_mat_lanczos_diagnostics_point(matvec, x, steps)`
- `jcb_mat_funm_action_arnoldi_point(matvec, x, dense_funm, steps, adjoint_matvec=...)`
- `jcb_mat_funm_action_arnoldi_with_diagnostics_point(matvec, x, dense_funm, steps, adjoint_matvec=...)`
- `jcb_mat_funm_action_hermitian_point(matvec, x, dense_funm, steps, adjoint_matvec=...)`
- `jcb_mat_funm_integrand_arnoldi_point(matvec, x, dense_funm, steps, adjoint_matvec=...)`
- `jcb_mat_funm_integrand_hermitian_point(matvec, x, dense_funm, steps, adjoint_matvec=...)`
- `jcb_mat_trace_integrand_point(matvec, x, adjoint_matvec=...)`
- `jcb_mat_funm_trace_integrand_arnoldi_point(matvec, x, scalar_fun, steps, adjoint_matvec=...)`
- `jcb_mat_trace_estimator_point(matvec, probes, adjoint_matvec=...)`
- `jcb_mat_trace_estimator_with_diagnostics_point(matvec, probes, adjoint_matvec=...)`
- `jcb_mat_logdet_slq_point(matvec, probes, steps, adjoint_matvec=...)`
- `jcb_mat_logdet_slq_with_diagnostics_point(matvec, probes, steps, adjoint_matvec=...)`
- `jcb_mat_logdet_solve_point(matvec, rhs, ...)`
- `jcb_mat_logdet_solve_basic(matvec, rhs, ...)`
- `jcb_mat_multi_shift_solve_point(matvec, rhs, shifts, hermitian=..., **kwargs)`
- `jcb_mat_multi_shift_solve_hermitian_point(matvec, rhs, shifts, **kwargs)`
- `jcb_mat_multi_shift_solve_hpd_point(matvec, rhs, shifts, **kwargs)`
- `jcb_mat_multi_shift_solve_basic(matvec, rhs, shifts, hermitian=..., **kwargs)`
- `jcb_mat_minres_solve_action_point(matvec, rhs, preconditioner=..., **kwargs)`
- `jcb_mat_minres_inverse_action_point(matvec, rhs, preconditioner=..., **kwargs)`
- `jcb_mat_eigsh_point(matvec, size=..., k=..., which=..., steps=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_with_diagnostics_point(...)`
- `jcb_mat_eigsh_restarted_point(matvec, size=..., k=..., which=..., steps=..., restarts=..., block_size=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_restarted_with_diagnostics_point(...)`
- `jcb_mat_eigsh_block_point(matvec, size=..., k=..., which=..., steps=..., block_size=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_block_with_diagnostics_point(...)`
- `jcb_mat_eigsh_krylov_schur_point(...)`
- `jcb_mat_eigsh_krylov_schur_with_diagnostics_point(...)`
- `jcb_mat_eigsh_davidson_point(matvec, size=..., k=..., which=..., subspace_iters=..., block_size=..., preconditioner=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_davidson_with_diagnostics_point(...)`
- `jcb_mat_eigsh_jacobi_davidson_point(matvec, size=..., k=..., which=..., subspace_iters=..., block_size=..., preconditioner=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_jacobi_davidson_with_diagnostics_point(...)`
- `jcb_mat_generalized_operator_plan_prepare(a_matvec, b_matvec, b_preconditioner=..., tol=..., atol=..., maxiter=...)`
- `jcb_mat_geigsh_point(a_matvec, b_matvec, size=..., k=..., which=..., steps=..., b_preconditioner=...)`
- `jcb_mat_geigsh_with_diagnostics_point(...)`
- `jcb_mat_eigsh_shift_invert_point(matvec, size=..., shift=..., k=..., which=..., steps=..., preconditioner=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_shift_invert_with_diagnostics_point(...)`
- `jcb_mat_generalized_shift_invert_operator_plan_prepare(a_matvec, b_matvec, shift=..., preconditioner=..., tol=..., atol=..., maxiter=...)`
- `jcb_mat_geigsh_shift_invert_point(a_matvec, b_matvec, size=..., shift=..., k=..., which=..., steps=..., preconditioner=...)`
- `jcb_mat_geigsh_shift_invert_with_diagnostics_point(...)`
- `jcb_mat_neigsh_point(matvec_builder, dmatvec_builder, size=..., lambda0=..., newton_iters=..., eig_steps=...)`
- `jcb_mat_neigsh_with_diagnostics_point(...)`
- `jcb_mat_peigsh_point(coeff_matvecs, size=..., lambda0=..., newton_iters=..., eig_steps=...)`
- `jcb_mat_peigsh_with_diagnostics_point(...)`
- `jcb_mat_eigsh_contour_point(matvec, size=..., center=..., radius=..., k=..., which=..., quadrature_order=..., block_size=..., preconditioner=..., adjoint_matvec=...)`
- `jcb_mat_eigsh_contour_with_diagnostics_point(...)`
- `jcb_mat_rademacher_probes_like(x, key=..., num=...)`
- `jcb_mat_normal_probes_like(x, key=..., num=...)`

## Current Methodology

Point:
- midpoint complex linear algebra
- result boxed back to an outward complex interval box

Basic:
- `matmul` / `matvec` use box arithmetic in canonical `(..., 4)` layout
- `solve_basic` currently uses midpoint solve plus outward boxing
- `triangular_solve_basic` currently uses midpoint triangular solve plus outward boxing
- `lu_basic` currently uses midpoint LU plus outward boxing of `P`, `L`, `U`

This means:
- `matmul_basic` and `matvec_basic` are genuine complex-box substrate operations
- `solve_basic` is currently a first substrate implementation, not yet a final rigorous box-linear-solve path

Matrix-free layer:
- Arnoldi is implemented in pure JAX with fixed-shape scan state and explicit projected orthogonalisation.
- The primary contract is action-first and operator-first, not dense matrix-function construction.
- restarted Krylov support now exists for the matrix-free `expm` action path via repeated scaled-action application
- block right-hand-side support now exists for the restarted `expm` action path through a batched matrix-free wrapper
- multi-shift solve support now exists as a shared operator-plan path for Hermitian/HPD and general shifted solves
- partial-spectrum eigensolver support now includes restarted, block, Davidson, Jacobi-Davidson, shift-invert, and contour-filter entry points; these now also have diagnostics surfaces, but still need deeper convergence-policy hardening
- generalized Hermitian-definite partial-spectrum support now exists through `jcb_mat_geigsh_*`, including a generalized shift-invert spectral-transform path that applies `(A - sigma B)^{-1} B`
- polynomial and nonlinear Hermitian point surfaces now exist through Newton refinement on the smallest-magnitude shift-invert eigenpair, with polynomial operators built automatically from coefficient operator families
- Hermitian projected action/integrand paths now have dedicated Lanczos-backed implementations, so the main Hermitian/HPD function-action wrappers no longer route through generic Arnoldi kernels by default
- `logdet` and solve can now be returned together through `jcb_mat_logdet_solve_*`, with shared auxiliary metadata packaged in `matrix_free_core`
- operator plans now also cover shell callbacks, finite-difference Jacobian-vector products, block/vblock sparse adapters, and a parameter-differentiable sparse `BCOO` closure path
- Backward support now exists for the input/probe vector pathways through custom VJPs on:
  - `jcb_mat_funm_action_arnoldi_point`
  - `jcb_mat_funm_integrand_arnoldi_point`
  - `jcb_mat_trace_integrand_point`
- Complex backward paths require `adjoint_matvec` whenever the operator is not self-adjoint.
- Estimator gradients flow through those custom VJPs, so `trace_estimator_point` and `logdet_slq_point` are probe-differentiable without tracing naively through the Arnoldi loop state.

## Diagnostics And Benchmarks

Current correctness coverage:
- [test_jcb_mat_chassis.py](/tests/test_jcb_mat_chassis.py)
- includes operator apply, polynomial action, `expm` action, Arnoldi exact diagonal cases, trace/logdet estimators, and backward probe-gradient checks

Current benchmark coverage:
- [benchmark_matrix_free_krylov.py](/benchmarks/benchmark_matrix_free_krylov.py)
- current report: [matrix_free_krylov_benchmark.md](/docs/status/reports/matrix_free_krylov_benchmark.md)

Current diagnostic contract:
- structured diagnostics now exist via `JcbMatKrylovDiagnostics`
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
- the current diagnostic surface also includes:
  - explicit `adjoint_matvec` threading for complex backward correctness
  - explicit forward-action vs backward-gradient timing in the benchmark runner
  - explicit test coverage for exact diagonal cases before using the same path in stochastic estimators
  - verified hot-path implementation remains inside JAX execution: no SciPy calls, no NumPy host kernels, no callback ops, and no Python loops in the Krylov core

## Not Yet Implemented / Not Yet Hardened

Planned matrix-function families:
- `jcb_mat_logm`
- `jcb_mat_sqrtm`
- `jcb_mat_rootm`
- `jcb_mat_signm`

Planned lower-level substrate:
- `qr`
- Hessenberg / Schur-compatible reductions

Still missing or incomplete on the current public matrix-free path:
- broader operator-parameter adjoints beyond the sparse parametric operator-plan path
- fully specified `basic` box-enclosure policy for stochastic estimators and solve-action families
- mature locking / deflation / convergence-history policy for the newer eigensolver tranche
- Hermitian-specialized tightening where generic Arnoldi-style machinery is still reused more than ideal

## Design Intent

- obey repo dtype, batching, and AD rules
- keep matrix substrate separate from the canonical Arb-like `acb_mat` namespace
- make the substrate reusable for matrix-free Arnoldi, stochastic trace estimation, and later RF77-facing operator workflows
