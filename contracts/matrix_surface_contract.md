Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Contract

## Scope

This contract covers the dense, sparse, and matrix-free matrix families in `arbplusjax`, including their execution-strategy split, JAX compilation expectations, AD expectations, and benchmark/report obligations.

## Families

The governed matrix families are:

- `arb_mat`: dense real interval matrices
- `acb_mat`: dense complex box matrices
- `srb_mat`: sparse real matrices
- `scb_mat`: sparse complex matrices
- `jrb_mat`: real matrix-free operators
- `jcb_mat`: complex matrix-free operators

Optional external matrix/eigensolver backends such as PETSc/SLEPc are not part of the governed JAX matrix-family ownership split. When present, they belong under `src/arbplusjax/backends/` and must not silently replace the JAX-governed `jrb_mat` / `jcb_mat` operator layer.

Block and variable-block sparse families follow the same sparse strategy rules when they expose analogous surfaces, including `matvec`, cached `matvec`, `rmatvec`, cached `rmatvec`, and operator-layer adapter coverage when those representations participate in matrix-free workflows.

## Execution strategy taxonomy

The stable execution-strategy vocabulary for matrix APIs is:

- `dense`: direct dense kernels over explicit matrix payloads
- `cached`: prepare/apply reuse path for repeated shape-stable calls
- `matvec`: operator application against one or more right-hand sides
- `rmatvec`: transpose or adjoint-style operator application
- `factorized`: explicit plan-backed solve or decomposition reuse path
- `operator_plan`: matrix-free prepared operator payload for repeated Krylov-style work

Algorithmic method selection and execution strategy are separate concerns. For example, Krylov restart count or stochastic logdet estimator choice is not the same axis as `cached` versus `operator_plan`.

## JAX compilation contract

- Matrix entry points are expected to remain pure-JAX and traceable.
- Repeated execution paths should prefer prepare/apply or plan/apply APIs instead of rebuilding equivalent traced payloads on every call.
- When a padded fixed-shape batch path exists, it is the preferred repeated-batch route for avoiding shape-driven recompiles.
- Cached plans and operator plans must be JAX pytrees when they are part of the supported repeated-execution surface.
- Reusable preconditioner plans must also be JAX pytrees when they participate in plan-native solve or eigensolver paths.
- Point-mode hot paths must remain the primary optimized execution engine. Wrapper layers must not force point-mode calls through avoidable boxing or non-shape-stable detours.
- Sparse families, including block and variable-block sparse families when they expose those routes, must treat `matvec`, cached `matvec`, `rmatvec`, and cached `rmatvec` as first-class optimized execution strategies rather than secondary compatibility helpers.
- Structured sparse symmetric/Hermitian point paths must not silently downgrade to dense reconstruction for symmetry checks, Cholesky, or LDL factorization.
- Sparse partial Hermitian/symmetric spectral routines such as `eigsh` must be owned by the matrix-free/operator layer and reached from sparse families through operator-plan delegation rather than dense midpoint reconstruction.
- Matrix-free solve and partial-spectrum surfaces may advertise reusable preconditioner-plan, multi-shift, and restarted/block Krylov variants, but those variants must remain plan-reuse friendly and benchmarked as compile-vs-execute claims rather than one-off wrappers.

## AD contract

- Public matrix entry points that are documented as differentiable must remain usable with JAX AD on their advertised payloads.
- Plan-backed paths may use specialized AD-safe wrappers when raw callable-oriented custom-VJP paths are not valid for traced plan payloads.
- When an operation is not yet AD-hardened, that limitation must be reflected in tests, contracts, or benchmark/workbook notes rather than hidden behind silent fallback.

## Batch contract

- Dense, sparse, and matrix-free families should expose batch-friendly entry points where repeated evaluation is a normal workload.
- Padded or fixed-shape batch helpers are preferred for benchmarked repeated-execution paths because they reduce recompilation churn.
- Batch helpers must preserve the same numerical semantics as their scalar or single-rhs counterparts.

## Coverage contract

- Each governed family must have correctness tests in `tests/`.
- Each governed family must appear in a benchmark surface under `benchmarks/`.
- The matrix workbook/report layer under `docs/reports/` must name the benchmark entry points and explain what is being compared.

## Sparse Basic Contract

For `srb_mat` and `scb_mat`, `basic` is a governed semantic mode for matrix operations, not a promise that every storage helper has an interval or box analogue.

The sparse `basic` contract covers the operational matrix surface:

- structural matrix queries and transforms used in computations
  - `to_dense`, `transpose`, `conjugate_transpose` where relevant
  - `symmetric_part` / `hermitian_part`
  - `is_symmetric` / `is_hermitian`
  - `is_spd` / `is_hpd`
  - `trace`, `norm_fro`, `norm_1`, `norm_inf`
- operator application surface
  - `matvec`
  - `rmatvec`
  - `matmul_dense_rhs`
  - cached `matvec` prepare/apply
  - cached `rmatvec` prepare/apply
  - governed fixed and padded batch helpers for the above operator paths
- factorization and solve surface
  - `triangular_solve`
  - `cho`
  - `ldl`
  - `solve`
  - LU solve-plan prepare/apply
  - SPD/HPD solve-plan prepare/apply
  - `solve_lu`
  - `solve_transpose`
  - `solve_add`
  - `solve_transpose_add`
  - `mat_solve`
  - governed fixed and padded batch helpers for the above solve paths
- higher matrix functions already exposed through sparse mode wrappers
  - `det`
  - `inv`
  - `sqr`
  - `charpoly`
  - `pow_ui`
  - `exp`
  - `eigvalsh`
  - `eigh`
  - `eigsh`

## Matrix-Free Plan Contract

For `jrb_mat` and `jcb_mat`, the governed matrix-free repeated-use surface includes:

- operator plans
  - dense operator plans
  - sparse operator plans
  - shell operator plans with explicit callback/context payloads
  - finite-difference Jacobian-vector operator plans with explicit base-point update
  - transpose / adjoint / `rmatvec` plan variants where applicable
- reusable preconditioner plans
  - identity plans
  - dense plans
  - diagonal plans
  - Jacobi-style plans derived from dense or sparse operator payloads
  - shell preconditioner plans
- shifted and recycled Krylov plan payloads
  - shifted solve plans for shared-operator multi-shift solves
  - recycled Krylov state pytrees for basis reuse
- repeated-use Krylov entry points
  - solve-action and inverse-action plan paths
  - symmetric / Hermitian indefinite `minres` plan paths
  - multi-shift solve plan paths
  - partial-spectrum `eigsh`
  - restarted / block partial-spectrum `eigsh`
  - future Krylov-Schur / Davidson / Jacobi-Davidson / contour eigensolver families when they are added as JAX-native operator capabilities

For this surface:

- plan-backed JIT wrappers must keep plan payloads dynamic and only true compile-shape or algorithm-configuration arguments static
- repeated-use claims must be benchmarked with separate compile-time and steady-state execute-time slices
- AD claims for plan-backed solve, inverse, determinant, and action paths must be reflected in tests or benchmark notes
- dense and sparse families may adapt into this surface, but they must not become independent owners of shell, finite-difference, or Krylov plan ecosystems
- SLEPc-inspired capability placement follows the same rule:
  - reusable spectral-transformation, restart/locking, correction-equation, and contour/quadrature substrate belongs in shared matrix-free core
  - public operator-facing solver functionality belongs in `jrb_mat` / `jcb_mat`
  - Krylov-Schur, Davidson, Jacobi-Davidson, contour eigensolvers, and spectral-transform eigensolver families must be implemented natively in JAX on that substrate rather than delegated to optional external backends
  - optional external SLEPc wrappers belong under `src/arbplusjax/backends/slepc/`

The following sparse surfaces are intentionally point-only unless explicitly promoted later:

- constructors and storage factories
  - `from_dense_*`
  - `coo`, `csr`, `bcoo`
- format conversions and raw storage bridges
  - `coo_to_csr`, `csr_to_coo`, `coo_to_bcoo`, `csr_to_bcoo`, `bcoo_to_coo`
  - raw `*_to_dense` storage helpers outside the governed mode wrapper surface

The following lightweight structural helpers may remain mode-addressable for API uniformity, but they are not the main sparse `basic` closure target:

- `shape`
- `nnz`
- `zero`
- `identity`
- `permutation_matrix`
- `diag`
- `diag_matrix`
- `submatrix`

Tests should enforce both sides of this contract:

- governed sparse matrix operations must have `basic` coverage through mode wrappers or explicit `*_basic` functions
- point-only constructor and conversion helpers must not accidentally grow `basic` mode wrappers without an explicit contract update

## Source of truth

- `src/arbplusjax/arb_mat.py`
- `src/arbplusjax/acb_mat.py`
- `src/arbplusjax/srb_mat.py`
- `src/arbplusjax/scb_mat.py`
- `src/arbplusjax/jrb_mat.py`
- `src/arbplusjax/jcb_mat.py`
- `docs/implementation/matrix_stack.md`
- `docs/implementation/external/slepc.md`
- `benchmarks/benchmark_dense_matrix_surface.py`
- `benchmarks/benchmark_sparse_matrix_surface.py`
- `benchmarks/benchmark_matrix_free_krylov.py`
- `benchmarks/benchmark_matrix_stack_diagnostics.py`
