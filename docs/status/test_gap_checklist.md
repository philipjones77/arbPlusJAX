Last updated: 2026-03-22T00:00:00Z

# Whole-Repo Test Gap Checklist

This is the execution checklist that sits underneath
[test_coverage_matrix.md](/docs/status/test_coverage_matrix.md).
It answers a narrower question:

- which runtime modules already have direct test owners
- which modules are only covered indirectly through higher-level surfaces
- which modules still need an explicit test owner

The goal is to make “test the whole repo” actionable.

## 1. Direct Test Owners Already Exist

These modules already have clearly named test owners in `tests/`.

### Core and scalar layers

- `acb_calc`
- `acb_core`
- `acf`
- `arb_calc`
- `arb_core`
- `arb_fmpz_poly`
- `arb_fpwrap`
- `arf`
- `fmpr`
- `fmpz_extras`
- `fmpzi`
- `mag`

### Dense and sparse matrix layers

- `acb_mat`
- `arb_mat`
- `bool_mat`
- `jcb_mat`
- `jrb_mat`
- `matfree_adjoints`
- `matrix_free_basic`
- `mp_mode`
- `scb_block_mat`
- `scb_mat`
- `scb_vblock_mat`
- `srb_block_mat`
- `srb_mat`
- `srb_vblock_mat`

### Special-function and transform layers

- `boost_hypgeom`
- `cubesselk`
- `dft`
- `double_gamma`
- `double_interval`
- `hypgeom`
- `nufft`
- `shahen_double_gamma`
- `stable_kernels`

### Analytic / algebraic / domain layers

- `acb_dirichlet`
- `acb_elliptic`
- `acb_modular`
- `acb_poly`
- `arb_poly`
- `bernoulli`
- `dirichlet`
- `dlog`
- `partitions`

### API / runtime / metadata layers

- `api`
- `capability_registry`
- `cusf_compat`
- `elementary`
- `function_provenance`
- `jax_diagnostics`
- `jax_precision`
- `point_wrappers`
- `public_metadata`
- `runtime`
- `soft_ops`

## 2. Indirect Coverage Exists, But Explicit Owners Are Weak

These modules are exercised through higher-level tests, but the repo would be
clearer if each eventually had a more explicit owner.

### Wrappers and routing helpers

- `ball_wrappers`
  Covered indirectly through API, mode, and point-wrapper surfaces.
- `baseline_wrappers`
  Covered indirectly through API and reference-facing surfaces.
- `calc_wrappers`
  Covered indirectly through `arb_calc` / `acb_calc` and API tests.
- `core_wrappers`
  Covered indirectly through scalar chassis and mode tests.
- `dft_wrappers`
  Covered indirectly through DFT chassis tests.
- `dirichlet_wrappers`
  Covered indirectly through Dirichlet chassis tests.
- `double_interval_wrappers`
  Covered indirectly through interval and mode tests.
- `hypgeom_wrappers`
  Covered indirectly through hypergeometric chassis and mode tests.
- `mat_wrappers`
  Covered indirectly through `test_mat_modes.py` and matrix chassis tests.
- `modular_elliptic_wrappers`
  Covered indirectly through elliptic/modular chassis tests.
- `poly_wrappers`
  Covered indirectly through polynomial chassis tests.
- `wrappers_common`
  Covered indirectly through all wrapper-driven surfaces.

### Shared matrix infrastructure

- `block_sparse_core`
  Covered indirectly through block/vblock sparse chassis tests.
- `iterative_solvers`
  Covered indirectly through matrix-free public surfaces.
- `kernel_helpers`
  Covered indirectly through stable-kernel and special-function surfaces.
- `krylov_solvers`
  Covered indirectly through `jrb_mat` / `jcb_mat`.
- `mat_common`
  Covered indirectly through dense matrix tests.
- `matrix_free_core`
  Covered indirectly through `jrb_mat`, `jcb_mat`, `matrix_free_basic`, and stack-contract tests.
- `sampling_helpers`
  Covered indirectly through stochastic/logdet paths.
- `sparse_common`
  Covered indirectly through sparse matrix chassis tests.
- `sparse_core`
  Covered indirectly through sparse matrix chassis tests.
- `transform_common`
  Covered indirectly through DFT/NUFFT surfaces.

### Metadata, support, and validation

- `checks`
  Covered indirectly through status and implementation tests.
- `coeffs`
  Covered indirectly through polynomial and analytic families.
- `precision`
  Covered indirectly through `jax_precision` tests and package import behavior.
- `soft_types`
  Covered indirectly through `soft_ops`.
- `validation`
  Covered indirectly through runtime/API correctness checks.

### Special-function support modules

- `bessel_kernels`
  Covered indirectly through Bessel/Hankel/spherical Bessel tests.
- `barnesg`
  Covered indirectly through Barnes/double-gamma-facing tests, but should have a clearer direct owner.

## 3. Modules That Still Need Explicit Test Owners

These are the places where the repo is most likely to benefit from adding a
focused test file instead of relying on incidental coverage.

### Highest priority

- `matrix_free_core`
  Add a focused shared-substrate contract test file for operator plans,
  restart helpers, generalized operator plans, and shape/static-arg behavior.
- `bessel_kernels`
  Add a direct low-level kernel contract file so cylindrical kernels are tested
  independently from higher-level spherical/Hankel wrappers.
- `barnesg`
  Add a direct Barnes G surface test instead of relying on adjacent
  double-gamma coverage.

### Next priority

- `block_sparse_core`
- `sparse_core`
- `sparse_common`
- `krylov_solvers`
- `iterative_solvers`
- `transform_common`
- `soft_types`
- `validation`

## 4. Test Files To Add

The first concrete additions should be:

- `tests/test_matrix_free_core_contracts.py`
- `tests/test_bessel_kernels_contracts.py`
- `tests/test_barnesg_contracts.py`

Then, if those are green:

- `tests/test_block_sparse_core_contracts.py`
- `tests/test_sparse_core_contracts.py`
- `tests/test_krylov_solvers_contracts.py`
- `tests/test_transform_common_contracts.py`

## 5. Execution Order

1. Add direct-owner tests for the six highest-priority modules above.
2. Re-run the CPU chassis/profile suite.
3. Expand AD and compile-behavior assertions where those new tests expose weak spots.
4. Add the second-priority shared-infra test files.
5. Re-run the full CPU harness, then opt-in parity and benchmark slices.
