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

- `barnesg`
- `bessel_kernels`
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

- `matrix_free_core`
- `block_sparse_core`
- `sparse_core`
- `iterative_solvers`
- `krylov_solvers`
- `transform_common`
- `kernel_helpers`
  Covered indirectly through stable-kernel and special-function surfaces.
- `mat_common`
  Covered indirectly through dense matrix tests.
- `sampling_helpers`
  Covered indirectly through stochastic/logdet paths.
- `sparse_common`
  Covered indirectly through sparse matrix chassis tests.

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

## 3. Modules That Still Need Explicit Test Owners

These are the places where the repo is most likely to benefit from adding a
focused test file instead of relying on incidental coverage.

### Highest priority

- `sparse_common`
  Add a direct sparse-common contract file for shared sparse dataclasses,
  batching, and canonicalization helpers.
- `iterative_solvers`
  Add a direct iterative-solver contract file for the non-Krylov wrapper layer
  used by sparse and matrix-free surfaces.
- `soft_types`
  Add a direct test owner for the lightweight dtype/container typing helpers.
- `validation`
  Add a direct test owner for shared validation helpers rather than relying only
  on runtime/API call sites.

### Next priority

- `sparse_common`
- `iterative_solvers`
- `soft_types`
- `validation`

## 4. Test Files To Add

The first concrete additions should be:

- `tests/test_sparse_common_contracts.py`
- `tests/test_iterative_solvers_contracts.py`
- `tests/test_soft_types_contracts.py`
- `tests/test_validation_contracts.py`

Then, if that is green:
- `tests/test_sparse_common_contracts.py`
- `tests/test_iterative_solvers_contracts.py`
- `tests/test_soft_types_contracts.py`
- `tests/test_validation_contracts.py`

## 5. Execution Order

1. Add direct-owner tests for the four highest-priority modules above.
2. Re-run the CPU chassis/profile suite.
3. Expand AD and compile-behavior assertions where those new tests expose weak spots.
4. Add the second-priority shared-infra test files.
5. Re-run the full CPU harness, then opt-in parity and benchmark slices.
