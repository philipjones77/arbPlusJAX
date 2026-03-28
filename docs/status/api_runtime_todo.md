Last updated: 2026-03-28T00:00:00Z

# API Runtime TODO

This file tracks public API, runtime, metadata, and validation backlog
separately from the mathematical-family backlogs.

Status legend:
- `done`: landed in code and covered at least by targeted tests
- `in_progress`: partially implemented or exposed, but still needs hardening
- `planned`: accepted roadmap item, not yet at implementation level

## API / Runtime / Metadata / Validation

Status: `in_progress`

- `done`
  - public metadata exists for family, stability, method tags, regime tags,
    and derivative status
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
  - keep
    [arb_mat_implementation.md](/docs/implementation/modules/arb_mat_implementation.md),
    [acb_mat_implementation.md](/docs/implementation/modules/acb_mat_implementation.md),
    [jrb_mat_implementation.md](/docs/implementation/modules/jrb_mat_implementation.md),
    and
    [jcb_mat_implementation.md](/docs/implementation/modules/jcb_mat_implementation.md)
    aligned with source

## Priority Test-Owner Additions

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

## Missing C Implementation Snapshot

Status: `in_progress`

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
belong in `docs/reports/missing_impls/`.
