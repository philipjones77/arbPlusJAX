Last updated: 2026-03-16T22:30:00Z

# Structured Matrix Functionality POA

## Purpose

This document is the plan-of-action reference for general matrix-library functionality in this workspace.

It is not limited to RandomFields77. It defines the capability map we should refer back to whenever new matrix functionality is requested, and it records where that functionality should land in the repository.

## External Reference Ordering

Use these external libraries as design references in this order:

1. STRUMPACK
   - primary reference for broad structured-matrix API design
2. HODLRlib
   - primary reference for HODLR solve/logdet/factor functionality
3. hmglib
   - primary reference for GPU-resident hierarchical construction/apply ideas

Interpretation:

- STRUMPACK informs the top-level interface breadth
- HODLRlib informs what a concrete hierarchical CPU backend must implement
- hmglib informs what a future hierarchical GPU backend should optimize

## Top-Level Capability Stack

This is the general matrix-library checklist for this repository.

### 1. Core Matrix Object Model

Required capabilities:

- shape
- dtype
- storage format
- device / backend location
- symmetry flags
- definiteness flags
- sparsity / structure flags
- mutability / view semantics

Matrix kinds:

- dense
- sparse
- diagonal
- banded
- block
- triangular
- symmetric / Hermitian
- positive definite
- permutation
- low-rank
- structured
- matrix-free / operator

Landing zone:

- runtime objects: `src/arbplusjax/*mat*.py`, `src/arbplusjax/sparse_common.py`, `src/arbplusjax/mat_common.py`
- object/spec docs: `docs/objects/`, `docs/specs/`
- public metadata / flags: `src/arbplusjax/public_metadata.py`

### 2. Construction

Required construction paths:

- from dense array
- from sparse triplets / CSR / CSC / COO
- from blocks
- from diagonals / bands
- from entry oracle
- from matrix-vector product oracle
- identity / zeros / ones
- random constructors
- structured constructors:
  - Toeplitz
  - circulant
  - Kronecker
  - Khatri-Rao
  - Hankel
  - Vandermonde
  - companion
  - hierarchical formats

Landing zone:

- dense explicit constructors: `arb_mat`, `acb_mat`
- sparse constructors: `srb_mat`, `scb_mat`, block variants
- matrix-free constructors: `jrb_mat`, `jcb_mat`
- future structured constructors: dedicated structured backend modules under `src/arbplusjax/`

### 3. Basic Algebra

Required capabilities:

- add
- subtract
- scale
- divide by scalar
- conjugate
- transpose
- adjoint
- scalar power where defined
- matrix-matrix multiply
- matrix-vector multiply
- transpose-multiply
- block multiply
- Hadamard product
- Kronecker product
- direct sum
- tensor product
- block assembly
- low-rank update
- rank-1 update
- shift by identity

Landing zone:

- dense kernels: `arb_mat`, `acb_mat`
- sparse kernels: `srb_mat`, `scb_mat`, block sparse modules
- operator/lazy kernels: `jrb_mat`, `jcb_mat`

### 4. Extraction, Views, and Conversions

Required capabilities:

- row / column / slice views
- diagonal extraction
- band extraction
- block extraction
- triangular part
- symmetric / antisymmetric part
- permutation views
- dense ↔ sparse conversions
- CSR ↔ CSC ↔ COO
- sparse ↔ block sparse
- explicit ↔ matrix-free
- host ↔ device
- precision conversion

Landing zone:

- runtime helpers: `src/arbplusjax/sparse_common.py`, `src/arbplusjax/mat_common.py`
- contracts: `contracts/`
- tests: `tests/test_*point_api.py`, chassis tests

### 5. Structural Queries and Diagnostics

Required capabilities:

- is square
- is symmetric
- is Hermitian
- is triangular
- is diagonal
- is banded
- is sparse
- is low-rank
- is block structured
- is SPD / PSD
- rank / estimated rank
- trace
- determinant / sign + logabsdet
- norms
- condition estimates
- bandwidth
- nnz
- memory footprint
- storage complexity
- eigenvalue bounds
- Gershgorin bounds
- spectral radius
- inertia
- extremal eigenvalues
- singular-value estimates

Landing zone:

- runtime queries: matrix modules
- optional diagnostics: dedicated opt-in utilities, not default hot paths
- docs: `docs/implementation/modules/`, `docs/specs/`
- benchmarks/notebooks: `benchmarks/`, `experiments/benchmarks/`

### 6. Factorizations

Required capabilities:

- LU
- QR
- LQ / RQ where relevant
- Cholesky
- LDLᵀ / LDLᴴ
- Schur
- Hessenberg
- tridiagonalization
- bidiagonalization
- SVD
- eigendecomposition
- sparse LU / Cholesky / LDLᵀ
- multifrontal / supernodal hooks
- symbolic + numeric factorization split
- structured factorizations:
  - banded
  - Toeplitz
  - Kronecker-aware
  - HODLR
  - HSS
  - H-matrix
  - BLR
  - low-rank / pivoted-Cholesky / CUR / ID
- update / downdate support

Landing zone:

- dense explicit: `arb_mat`, `acb_mat`
- sparse explicit: `srb_mat`, `scb_mat`
- structured/hierarchical future: new backend modules
- matrix-free surrogates and stochastic functionals: `jrb_mat`, `jcb_mat`

### 7. Solvers

Required capabilities:

- direct solve
- multi-RHS solve
- triangular solve
- transpose / adjoint solve
- SPD-specialized solve
- least squares
- saddle-point / constrained solve hooks
- iterative solvers:
  - CG / PCG
  - MINRES
  - GMRES
  - BiCGSTAB
  - LSQR / LSMR
  - block Krylov methods
- preconditioners:
  - Jacobi
  - ILU / incomplete Cholesky
  - block preconditioners
  - multigrid interfaces
  - hierarchical / low-rank corrections

Landing zone:

- iterative solvers: `src/arbplusjax/iterative_solvers.py`, `src/arbplusjax/krylov_solvers.py`
- dense/sparse solve entry points: matrix modules
- matrix-free solves and adjoints: `jrb_mat`, `jcb_mat`, `matfree_adjoints.py`

### 8. Spectral and Decomposition Methods

Required capabilities:

- full eigendecomposition
- partial eigenpairs
- generalized eigenproblems
- shift-invert support
- interior eigenvalue hooks
- block eigensolvers
- full and truncated SVD
- randomized SVD
- Lanczos
- Arnoldi
- block Lanczos
- subspace iteration
- LOBPCG
- Davidson / Jacobi-Davidson hooks

Landing zone:

- dense explicit spectral code: `arb_mat`, `acb_mat`
- matrix-free spectral code: `jrb_mat`, `jcb_mat`, `matfree_adjoints.py`

### 9. Determinants, Traces, and Matrix Functions

Required capabilities:

- determinant
- logdet
- sign/logabsdet
- determinant from factorization
- determinant updates
- trace
- trace of products
- trace of inverse-times-matrix
- stochastic trace estimation
- diagonal estimation
- Hutchinson / Hutch++ / XTrace-like estimators
- matrix exponential
- matrix logarithm
- matrix square root
- inverse square root
- fractional powers
- sign function
- resolvent
- rational / contour / polynomial matrix-function interfaces
- actions:
  - `exp(A)v`
  - `log(A)v`
  - `A^alpha v`
  - `f(A)v`

Landing zone:

- dense explicit matrix functions: `arb_mat`, `acb_mat`, `jrb_mat`, `jcb_mat`
- stochastic/operator-first functionals: `jrb_mat`, `jcb_mat`
- future structured logdet/trace backends: structured backend modules

### 10. Matrix-Free / Operator Layer

Required capabilities:

- shape
- dtype
- matvec
- matmat
- transpose matvec
- adjoint matvec
- diagonal action if available
- block / batched action
- composition
- sum / product of operators
- shift / scale
- lazy sum
- lazy product
- lazy Kronecker
- lazy block matrix
- lazy transpose / adjoint
- approximate inverse wrapper

Landing zone:

- primary home: `jrb_mat`, `jcb_mat`
- adjoint hooks: `matfree_adjoints.py`
- sparse operator reuse: `sparse_common.py`

### 11. Structured Matrix Layer

Required capabilities:

- diagonal / tridiagonal / banded
- Toeplitz / circulant / block Toeplitz
- Hankel / Vandermonde / Cauchy / companion
- low-rank `UV^T`
- CUR / ID / Nyström / randomized low-rank
- hierarchical formats:
  - HODLR
  - HSS
  - H-matrix
  - H²
  - BLR
  - butterfly / FMM-like
- structured compression / recompression
- structured matvec / solve / factor / logdet / diagnostics

Landing zone:

- future dedicated backend family, likely:
  - `DenseStructuredBackend`
  - `HierarchicalCPUBackend`
  - `HierarchicalGPUBackend`
- docs/specs first: `docs/specs/`

### 12. Sparse Matrix Layer

Required capabilities:

- COO / CSR / CSC / BSR / DIA
- ELL / HYB hooks for GPU contexts
- sparse matvec / matmat
- sparse addition / transpose / slicing
- graph-based utilities
- ordering / fill-reduction / symbolic sparsity support

Landing zone:

- `srb_mat`, `scb_mat`, block sparse variants
- `sparse_common.py`
- future ordering/graph helpers under `src/arbplusjax/`

### 13. Stability, Precision, and Error Control

Required capabilities:

- float32 / float64 / complex64 / complex128
- mixed precision
- arbitrary precision hooks
- iterative refinement
- residual norms
- backward error
- condition estimates
- convergence diagnostics
- orthogonality diagnostics
- truncation / approximation error estimates
- pivoting strategies
- jitter / regularization
- breakdown handling
- fallback algorithms

Landing zone:

- precision helpers: `jax_precision.py`
- core mode behavior: wrappers and matrix modules
- diagnostics contracts: docs/specs/contracts

### 14. Performance and Systems Functionality

Required capabilities:

- multithreading
- SIMD / vectorization
- GPU kernels
- multi-GPU and distributed hooks
- out-of-core hooks
- batched kernels
- workspace queries
- memory pool integration
- lazy materialization
- backend abstraction

Landing zone:

- benchmark harnesses: `benchmarks/`, `tools/run_test_harness.py`
- optional diagnostics and profiling: harness/notebook utilities, not runtime hot path
- backend-specific future modules

### 15. Differentiation and Sensitivity

Required capabilities:

- differentiable matvec
- differentiable solve
- differentiable Cholesky / QR / SVD where supported
- differentiable eigensolvers where supported
- differentiable logdet
- differentiable inverse quadratic form
- implicit differentiation interfaces
- JVP / VJP support
- derivative of determinant / logdet
- derivative through iterative solvers
- custom adjoint hooks

Landing zone:

- `matfree_adjoints.py`
- `jrb_mat`, `jcb_mat`
- future structured backend adjoints
- benchmarks/notebooks for AD validation under `experiments/benchmarks/`

### 16. Probabilistic / Statistics-Facing Interfaces

Required capabilities:

- `inv_quad(y)`
- `logdet()`
- `inv_quad_logdet(y)`
- Gaussian sampling
- whitening / coloring
- covariance / precision operator abstractions
- stochastic trace and diagonal estimators

Landing zone:

- operator-first matrix layer
- future RF77-facing wrappers and domain-aware operator layer

### 17. Interoperability, I/O, and Inspection

Required capabilities:

- serialization
- Matrix Market support
- NumPy / SciPy / MATLAB / Julia interchange
- factorization serialization
- pretty-printing
- summary reports
- structural visualization hooks
- sparsity / spectrum plotting
- profiling hooks
- logging / diagnostics

Landing zone:

- tools and docs
- `outputs/`, `results/`, benchmark/report generators

### 18. Testing and Benchmarking

Required capabilities:

- residual checks
- reconstruction checks
- orthogonality checks
- PD checks
- storage-format consistency
- randomized property tests
- runtime benchmarks
- memory benchmarks
- scaling tests
- backend comparisons
- accuracy/performance tradeoff sweeps
- four-mode tests and benchmarks where a four-mode surface exists

Landing zone:

- tests: `tests/`
- benchmark runners: `benchmarks/`
- broader exploratory sweeps: `experiments/benchmarks/`
- reports: `docs/status/reports/`

## Minimal Serious Matrix Library Checklist

This is the compressed must-have list we should treat as the minimum serious target:

- dense and sparse matrix objects
- matrix-free operator object
- matrix/vector multiply and transpose/adjoint multiply
- addition, scaling, block assembly
- format conversion
- structural predicates and metadata
- LU, QR, Cholesky, SVD, eigensolver
- direct and iterative solve
- determinant / logdet
- trace and norms
- low-rank approximation
- preconditioners
- sparse formats
- structured matrices
- partial eigensolvers
- matrix-function actions
- batched / GPU / parallel support
- diagnostics and error estimates
- operator abstraction
- randomized algorithms
- autodiff hooks
- serialization / interoperability

## RandomFields77-Specific Gap Layer

These are the pieces that the external structured-matrix libraries do not solve for this workspace:

- JAX tracing compatibility
- custom VJPs for solve, logdet, and inverse-quadratic forms
- observation-operator composition
- manifold / mesh / product-domain aware operators
- NumPy and JAX dual backends
- likelihood-centered interfaces:
  - `inv_quad(y)`
  - `logdet()`
  - `inv_quad_logdet(y)`
  - `sample(key)`
  - `trace(A^{-1} dA/dtheta)`

These should remain explicit roadmap items rather than being assumed to fall out automatically from an imported backend.

## How To Use This POA Later

When a new matrix feature is requested:

1. Map it to one or more sections in this POA.
2. State the target landing zone in the repo.
3. State whether it belongs in:
   - dense explicit matrix layer
   - sparse explicit matrix layer
   - matrix-free/operator layer
   - structured backend layer
   - probabilistic/RF77-facing layer
4. State whether it is:
   - minimum baseline
   - factorization/solve extension
   - logdet/trace extension
   - backend specialization
   - differentiation/adjoint work

This document is the canonical reference for that classification.
