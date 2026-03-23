Last updated: 2026-03-22T00:00:00Z

# Test Coverage Matrix

## Purpose

This document breaks the repo into the main functionality categories that need
testing, the helper/infrastructure categories that also need testing, the main
current test families, and the remaining gaps.

This matrix is not just a test-planning artifact. It is the default
organization lens for the repo.

Use these top-level categories as the primary way to interpret:

- implementation status
- report structure
- example notebook ownership
- benchmark grouping
- active TODO planning
- tranche-by-tranche production-readiness work

When a report, status page, benchmark summary, notebook inventory, or roadmap
needs a top-level grouping, it should default back to the categories defined in
this document unless there is a strong reason to use a narrower local view.

The repo testing model is:

- `tests/` verifies implementation correctness, contracts, AD behavior, and integration
- `benchmarks/` measures speed, compile cost, recompile behavior, backend comparison, and bottlenecks

## Top-Level Test Categories

These categories are also the default repo functionality categories.

### 1. Core Numeric Scalars

Scope:

- core Arb-like real/complex scalar functionality
- float wrappers and integer helper surfaces
- elementary numerical kernels and low-level wrappers

Main modules:

- `arb_core`, `acb_core`
- `arf`, `acf`, `fmpr`, `fmpzi`
- `arb_fpwrap`

Main tests:

- [test_arb_core_chassis.py](/tests/test_arb_core_chassis.py)
- [test_acb_core_chassis.py](/tests/test_acb_core_chassis.py)
- [test_arf_chassis.py](/tests/test_arf_chassis.py)
- [test_acf_chassis.py](/tests/test_acf_chassis.py)
- [test_fmpr_chassis.py](/tests/test_fmpr_chassis.py)
- [test_fmpzi_chassis.py](/tests/test_fmpzi_chassis.py)
- parity companions in `test_*_parity.py`

### 2. Interval / Box / Precision Modes

Scope:

- interval/box arithmetic
- mode wrappers
- `point` / `basic` / precision behavior
- precision-routing and dtype policy

Main modules:

- `double_interval`
- `point_wrappers`
- `mat_wrappers`
- `jax_precision`

Main tests:

- [test_double_interval_chassis.py](/tests/test_double_interval_chassis.py)
- [test_point_wrappers_contracts.py](/tests/test_point_wrappers_contracts.py)
- [test_mat_modes.py](/tests/test_mat_modes.py)
- [test_basic_wrappers.py](/tests/test_basic_wrappers.py)
- [test_mp_mode.py](/tests/test_mp_mode.py)
- [test_jax_precision.py](/tests/test_jax_precision.py)

### 3. Dense Matrix Functionality

Scope:

- dense real/complex matrix kernels
- cached dense apply / `rmatvec`
- factorization-backed solves
- structured dense helpers and aliases

Main modules:

- `arb_mat`
- `acb_mat`

Main tests:

- [test_arb_mat_chassis.py](/tests/test_arb_mat_chassis.py)
- [test_acb_mat_chassis.py](/tests/test_acb_mat_chassis.py)
- [test_dense_broad_surface.py](/tests/test_dense_broad_surface.py)
- [test_dense_plan_modes.py](/tests/test_dense_plan_modes.py)
- [test_dense_structured_modes.py](/tests/test_dense_structured_modes.py)
- [test_dense_eigh_and_aliases.py](/tests/test_dense_eigh_and_aliases.py)

### 4. Sparse / Block-Sparse / VBlock Functionality

Scope:

- sparse real/complex storage kernels
- COO / CSR / BCOO surface
- block-sparse and variable-block sparse adapters
- sparse interval/box storage wrappers

Main modules:

- `srb_mat`, `scb_mat`
- `srb_block_mat`, `scb_block_mat`
- `srb_vblock_mat`, `scb_vblock_mat`

Main tests:

- [test_srb_mat_chassis.py](/tests/test_srb_mat_chassis.py)
- [test_scb_mat_chassis.py](/tests/test_scb_mat_chassis.py)
- [test_srb_block_mat_chassis.py](/tests/test_srb_block_mat_chassis.py)
- [test_scb_block_mat_chassis.py](/tests/test_scb_block_mat_chassis.py)
- [test_srb_vblock_mat_chassis.py](/tests/test_srb_vblock_mat_chassis.py)
- [test_scb_vblock_mat_chassis.py](/tests/test_scb_vblock_mat_chassis.py)
- [test_sparse_point_api.py](/tests/test_sparse_point_api.py)
- [test_sparse_format_modes.py](/tests/test_sparse_format_modes.py)
- [test_sparse_structured_surface.py](/tests/test_sparse_structured_surface.py)
- [test_sparse_basic_contracts.py](/tests/test_sparse_basic_contracts.py)

### 5. Matrix-Free / Operator Functionality

Scope:

- operator plans
- shell and finite-difference plans
- solve / inverse actions
- `logdet` / `det`
- multi-shift solves
- eigensolvers, generalized eigensolvers, polynomial/nonlinear eigensolvers
- diagnostics and AD-facing matrix-free behavior

Main modules:

- `matrix_free_core`
- `matrix_free_basic`
- `jrb_mat`
- `jcb_mat`

Main tests:

- [test_jrb_mat_chassis.py](/tests/test_jrb_mat_chassis.py)
- [test_jcb_mat_chassis.py](/tests/test_jcb_mat_chassis.py)
- [test_jrb_mat_logdet_contracts.py](/tests/test_jrb_mat_logdet_contracts.py)
- [test_jrb_mat_selected_inverse.py](/tests/test_jrb_mat_selected_inverse.py)
- [test_matrix_free_basic.py](/tests/test_matrix_free_basic.py)
- [test_matrix_stack_contracts.py](/tests/test_matrix_stack_contracts.py)
- [test_matfree_adjoints.py](/tests/test_matfree_adjoints.py)

### 6. Special Functions

Scope:

- special-function kernels
- hypergeometric families in canonical and alternative lineages
- incomplete/gamma/Bessel/Hankel stacks
- hardening and containment-focused special-function work

Main modules:

- `hypgeom`
- `boost_hypgeom`
- `gamma`
- incomplete special-function modules
- tail-acceleration scaffolding

Main tests:

- [test_hypgeom_chassis.py](/tests/test_hypgeom_chassis.py)
- [test_hypgeom_modes_complete.py](/tests/test_hypgeom_modes_complete.py)
- [test_incomplete_gamma.py](/tests/test_incomplete_gamma.py)
- [test_incomplete_bessel_i.py](/tests/test_incomplete_bessel_i.py)
- [test_incomplete_bessel_k.py](/tests/test_incomplete_bessel_k.py)
- [test_gamma_hardening.py](/tests/test_gamma_hardening.py)
- [test_hankel_special.py](/tests/test_hankel_special.py)
- [test_spherical_bessel_special.py](/tests/test_spherical_bessel_special.py)
- [test_tail_acceleration_scaffold.py](/tests/test_tail_acceleration_scaffold.py)
- [test_boost_hypgeom.py](/tests/test_boost_hypgeom.py)
- [test_boost_ref_adapter.py](/tests/test_boost_ref_adapter.py)

Hypergeometric detail that needs explicit testing:

- canonical hypergeometric kernels:
  - `0F1`
  - `1F1`
  - `pFq`
  - complex-parameter and complex-argument paths
- alternative/reference hypergeometric kernels:
  - Boost-lineage `0F1`
  - Boost-lineage `1F1`
  - Boost-lineage `pFq`
  - `1F0` and `2F0` helper/reference surfaces where exposed
- mode coverage:
  - `point`
  - `basic`
  - precision-routed calls with `prec_bits` / `dps`
- batch coverage:
  - fixed batch entrypoints
  - padded batch entrypoints
  - mode-aware batch wrappers
- contract coverage:
  - shape propagation
  - dtype/complex promotion
  - parameter broadcasting
  - regularized vs non-regularized branches where supported
- parity/reference coverage:
  - canonical vs Boost comparison where both are defined
  - high-precision/Arb-style reference checks on small hard cases
- containment/hardening coverage:
  - cancellation-sensitive regions
  - large-argument or large-parameter cases
  - tail/asymptotic fallback scaffolding where wired into the family

Gamma-family detail that needs explicit testing:

- canonical gamma-family kernels:
  - `gamma`
  - `rgamma`
  - `lgamma` / `loggamma`
  - `digamma`
  - `polygamma`
- incomplete gamma surfaces:
  - lower / upper
  - regularized and non-regularized branches
  - parameter and argument derivative surfaces
- Barnes/double-gamma style families where exposed:
  - Barnes `G`
  - double-gamma / multiple-gamma style surfaces
  - log-form variants when provided
- contract coverage:
  - reflection / recurrence identities where applicable
  - real and complex argument support
  - dtype promotion and branch-cut handling
- parity/reference coverage:
  - JAX/Scipy/Arb reference checks on small and delicate cases
  - consistency between direct and log-form evaluations
- containment/hardening coverage:
  - poles and near-pole neighborhoods
  - large-argument asymptotics
  - cancellation around reflection and complement identities

Bessel/Hankel detail that needs explicit testing:

- cylindrical Bessel families:
  - `J`
  - `Y`
  - `I`
  - `K`
- spherical and modified spherical Bessel families:
  - spherical `j`
  - spherical `y`
  - modified spherical `i`
  - modified spherical `k`
- incomplete Bessel families where exposed:
  - incomplete `I`
  - incomplete `K`
- Hankel families:
  - Hankel first kind
  - Hankel second kind
- derivative and recurrence coverage:
  - explicit derivative helpers
  - three-term recurrences
  - half-integer identities
- method coverage:
  - series
  - recurrence
  - asymptotic / auto-selection paths
- contract coverage:
  - order/argument broadcasting
  - real vs complex argument behavior
  - branch and normalization conventions
- parity/reference coverage:
  - comparison to cylindrical/spherical identities
  - external reference checks on small hard cases
- containment/hardening coverage:
  - small-argument singular behavior
  - large-order or large-argument stability
  - oscillatory and cancellation-sensitive regions

### 7. Transforms / FFT / NUFFT

Scope:

- transform kernels
- internal NUFFT layer
- transform API correctness and batching

Main modules:

- FFT/NUFFT and related transform helpers

Main tests:

- [test_dft_chassis.py](/tests/test_dft_chassis.py)
- [test_nufft.py](/tests/test_nufft.py)

### 8. API / Facade / Metadata

Scope:

- public API routing
- metadata and provenance
- package-root export behavior

Main modules:

- `api`
- `public_metadata`
- `capability_registry`
- provenance/reporting helpers

Main tests:

- [test_api_metadata.py](/tests/test_api_metadata.py)
- [test_api_selection_contracts.py](/tests/test_api_selection_contracts.py)
- [test_function_provenance_reports.py](/tests/test_function_provenance_reports.py)
- [test_package_repo_naming.py](/tests/test_package_repo_naming.py)
- [test_all_functions_smoke.py](/tests/test_all_functions_smoke.py)

### 9. Analytic / Algebraic / Number-Theoretic Families

Scope:

- analytic function families outside the core special-function bucket
- Dirichlet, elliptic, modular, polynomial, and combinatorial helper surfaces
- repo-specific analytic kernels and domain-oriented wrappers

Main modules:

- `arb_calc`, `acb_calc`
- `dirichlet`, `acb_dirichlet`
- `acb_elliptic`, `acb_modular`
- `arb_poly`, `acb_poly`, `arb_fmpz_poly`
- `bernoulli`, `partitions`, `dlog`
- `cubesselk`, `stable_kernels`

Main tests:

- [test_arb_calc_chassis.py](/tests/test_arb_calc_chassis.py)
- [test_acb_calc_chassis.py](/tests/test_acb_calc_chassis.py)
- [test_dirichlet_chassis.py](/tests/test_dirichlet_chassis.py)
- [test_acb_dirichlet_chassis.py](/tests/test_acb_dirichlet_chassis.py)
- [test_acb_elliptic_chassis.py](/tests/test_acb_elliptic_chassis.py)
- [test_acb_modular_chassis.py](/tests/test_acb_modular_chassis.py)
- [test_arb_poly_chassis.py](/tests/test_arb_poly_chassis.py)
- [test_acb_poly_chassis.py](/tests/test_acb_poly_chassis.py)
- [test_arb_fmpz_poly_chassis.py](/tests/test_arb_fmpz_poly_chassis.py)
- [test_bernoulli_chassis.py](/tests/test_bernoulli_chassis.py)
- [test_partitions_chassis.py](/tests/test_partitions_chassis.py)
- [test_dlog_chassis.py](/tests/test_dlog_chassis.py)
- [test_cubesselk_api.py](/tests/test_cubesselk_api.py)
- [test_stable_kernels.py](/tests/test_stable_kernels.py)
- parity companions in `test_*_parity.py`

### 10. Compatibility / Soft Ops / Support Kernels

Scope:

- compatibility shims
- soft-type and soft-op layers
- boolean/support matrix helpers
- low-level support surfaces that still need direct implementation validation

Main modules:

- `cusf_compat`
- `soft_ops`, `soft_types`
- `bool_mat`
- `mag`
- `checks`, `validation`

Main tests:

- [test_cusf_compat.py](/tests/test_cusf_compat.py)
- [test_soft_ops.py](/tests/test_soft_ops.py)
- [test_bool_mat_chassis.py](/tests/test_bool_mat_chassis.py)
- [test_mag_chassis.py](/tests/test_mag_chassis.py)
- [test_custom_core_status.py](/tests/test_custom_core_status.py)
- [test_point_core_status.py](/tests/test_point_core_status.py)
- [test_elementary_layer.py](/tests/test_elementary_layer.py)

### 11. Runtime / Diagnostics / Harness

Scope:

- diagnostics payloads
- runtime manifest capture
- benchmark taxonomy and smoke harnesses

Main modules:

- runtime helpers
- diagnostics helpers
- benchmark taxonomy/harness files

Main tests:

- [test_jax_diagnostics.py](/tests/test_jax_diagnostics.py)
- [test_runtime_manifest.py](/tests/test_runtime_manifest.py)
- [test_benchmark_taxonomy.py](/benchmarks/test_benchmark_taxonomy.py)
- [test_benchmark_smoke.py](/benchmarks/test_benchmark_smoke.py)

### 12. Parity / External Reference Validation

Scope:

- Arb/FLINT reference parity
- Boost and other explicit reference comparisons

Main tests:

- all `test_*_parity.py`
- [test_boost_hypgeom.py](/tests/test_boost_hypgeom.py)
- [test_boost_ref_adapter.py](/tests/test_boost_ref_adapter.py)

## Helper / Infrastructure Categories That Also Need Tests

These are easy to under-test but should remain explicit:

- layout coercion helpers
  - dense/vector/rhs coercers
  - sparse interval/box converters
  - operator vector normalization
- cached-plan builders and apply helpers
- shell-plan and finite-difference-plan builders
- preconditioner-plan helpers
- dtype/x64 policy helpers
- diagnostics metadata attachment helpers
- API metadata/report generation helpers
- benchmark taxonomy and CLI smoke helpers
- source-tree bootstrap helpers used by scripts

Current concentrated coverage:

- [test_matrix_stack_contracts.py](/tests/test_matrix_stack_contracts.py)
- [test_dense_plan_modes.py](/tests/test_dense_plan_modes.py)
- [test_sparse_basic_contracts.py](/tests/test_sparse_basic_contracts.py)
- [test_jax_precision.py](/tests/test_jax_precision.py)
- [test_api_selection_contracts.py](/tests/test_api_selection_contracts.py)
- [test_benchmark_taxonomy.py](/benchmarks/test_benchmark_taxonomy.py)

## Coverage Dimensions Every Category Should Hit

For each major category, aim to cover:

1. correctness
   - known-value cases
   - residual checks
   - exact diagonal/identity/small dense truth where possible
2. contracts
   - shapes
   - dtypes
   - return structure
   - export/routing behavior
3. runtime
   - JIT-safe hot paths
   - stable-shape repeated-use paths
   - no accidental dense fallback where forbidden
4. AD
   - gradients where claimed
   - custom VJP/JVP paths where claimed
5. integration
   - adapters between dense/sparse/operator layers
   - runtime/harness/CLI entrypoints

## Current High-Priority Gaps

The main remaining test gaps are:

- broader operator-parameter AD coverage outside the current matrix-free sparse/dense/shell-plan slice
- more structured HPD/SPD gradient coverage across plan-backed determinant and solve surfaces
- broader matrix-free `basic` semantic coverage family-by-family
- more explicit compile/recompile behavior assertions for the hottest matrix-free paths
- more benchmark-oracle coverage around optional PETSc/SLEPc availability states if those environments become available in CI or scheduled runs

## Testing Work Plan

### Phase 1: Coverage Reconciliation

- verify every top-level category above has at least one current chassis or contract owner
- verify every public export family lands in one of:
  - chassis
  - focused contract
  - parity/reference
  - benchmark-only comparison
- record uncovered exports before adding new tests

### Phase 2: Correctness Expansion

- dense/sparse/matrix:
  - add residual and structured-branch checks where missing
- special functions:
  - add per-family hard-case and branch/fallback coverage
- analytic/algebraic families:
  - add exact identities, recurrence checks, and parity anchors where available

### Phase 3: Contract Expansion

- add explicit tests for:
  - broadcasting
  - dtype promotion
  - shape propagation
  - fixed vs padded batch surfaces
  - metadata/provenance routing

### Phase 4: AD Expansion

- extend gradient/JVP/VJP coverage for:
  - parameterized dense/sparse operator adapters
  - matrix-free solve/logdet/eigensolver families where gradients are claimed
  - derivative-exposing special-function families

### Phase 5: Runtime and Compile Behavior

- benchmark cold vs warm vs recompile for:
  - dense apply
  - sparse apply
  - matrix-free apply
  - `logdet`
  - eigensolvers
  - special-function batch entrypoints
- capture Python-loop-heavy paths and closure churn in benchmark artifacts

### Phase 6: Parity and External References

- run Arb/FLINT parity behind `ARBPLUSJAX_RUN_PARITY=1`
- expand Boost/reference overlap checks for special-function families
- keep PETSc/SLEPc and similar systems as benchmark/oracle comparison only

### Phase 7: Full Harness Execution

- CPU:
  - `smoke`
  - `chassis`
  - `matrix`
  - `special`
- GPU:
  - benchmark smoke
  - targeted performance slices with cold/warm split
- parity:
  - dedicated opt-in run after CPU chassis is green

## Execution Rule

“Test everything” in this repo should mean:

- every functionality category above has at least one canonical chassis/contract test family
- every helper/infrastructure category above has at least one explicit test owner
- full performance and bottleneck analysis stays in `benchmarks/`, not in ordinary correctness tests
