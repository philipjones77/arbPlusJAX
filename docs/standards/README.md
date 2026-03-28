Last updated: 2026-03-25T00:00:00Z

# Standards

This section holds repo-wide standards and governance-linked policy documents.

The detailed standards overlap in a few places, so the practical reading model is concept-first rather than filename-first. The current standards set consolidates into six primary concept groups.

## Consolidated Concept Groups

### 1. Runtime, Numerics, and Production Calling

Primary owner:
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)

Specialized companion documents:
- [engineering_standard.md](/docs/standards/engineering_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)
- [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [precision_standard.md](/docs/standards/precision_standard.md)
- [configuration_standard.md](/docs/standards/configuration_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)

Consolidation note:
- treat `jax_api_runtime_standard.md` as the canonical runtime/API contract
- treat `engineering_standard.md` as the hardening and status-interpretation overlay
- treat `caching_recompilation_standard.md` as the explicit cache, binder-reuse, prepared-plan, and recompilation-discipline companion
- treat `implicit_adjoint_operator_solve_standard.md` as the operator-first solve, transpose-solve, and implicit-adjoint differentiation companion
- treat `lazy_loading_standard.md` as the canonical import-time load and public lazy-boundary companion
- treat `configuration_standard.md` as the checked-in runtime/optional-backend configuration companion
- treat `core_scalar_service_calling_standard.md` as a tranche-specific specialization, not a second general runtime policy
- API calling shape, binder reuse, diagnostics payloads, logging hooks, optional backend declaration, and the rule that diagnostics/profiling stay outside the mandatory numeric hot path all belong to this runtime concept

### 2. Validation, Benchmarking, and Executable Examples

Primary owners:
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)

Specialized companion documents:
- [benchmark_grouping_standard.md](/docs/standards/benchmark_grouping_standard.md)
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)

Consolidation note:
- `benchmark_validation_policy_standard.md` owns measurement and benchmark-contract policy
- `benchmark_grouping_standard.md` is the taxonomy companion for the same benchmark concept
- `example_notebook_standard.md` is the executable-teaching analogue of the same validation/communication layer

### 3. Portability and Run Layout

Primary owners:
- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)
- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)

Consolidation note:
- these documents jointly own where things run and where artifacts live
- GitHub submission, Windows, Linux/WSL, and Colab portability expectations belong here rather than in ad hoc runbook notes

### 4. Contracts And Provider Boundary

Primary owner:
- [contract_and_provider_boundary_standard.md](/docs/standards/contract_and_provider_boundary_standard.md)

Consolidation note:
- this document consolidates the missing contract-placement and provider-boundary policy into one public-surface concept
- downstream-facing API capability contracts, metadata guarantees, and provider-grade surface rules belong here rather than in ad hoc per-family notes

### 5. Documentation Outputs And Generated Communication Surfaces

Primary owners:
- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [report_standard.md](/docs/standards/report_standard.md)
- [status_standard.md](/docs/standards/status_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)

Consolidation note:
- `generated_documentation_standard.md` owns the shared generation rule
- `repo_standards.md` owns repo-root communication and placement
- the report/status standards remain specialized audience documents under the same documentation-output concept

### 6. Theory, Notation, and Naming Semantics

Primary owners:
- [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)
- [function_naming_standard.md](/docs/standards/function_naming_standard.md)

Specialized companion documents:
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)

Consolidation note:
- `theory_notation_standard.md` owns methodology-note and notation governance
- function naming and test naming remain the explicit naming-policy layer

## Detailed Standards
- [benchmark_grouping_standard.md](/docs/standards/benchmark_grouping_standard.md)
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [configuration_standard.md](/docs/standards/configuration_standard.md)
- [contract_and_provider_boundary_standard.md](/docs/standards/contract_and_provider_boundary_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)
- [engineering_standard.md](/docs/standards/engineering_standard.md)
- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)
- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)
- [function_naming_standard.md](/docs/standards/function_naming_standard.md)
- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [metadata_registry_standard.md](/docs/standards/metadata_registry_standard.md)
- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
- [point_surface_standard.md](/docs/standards/point_surface_standard.md)
- [precision_standard.md](/docs/standards/precision_standard.md)
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)
- [release_execution_checklist_standard.md](/docs/standards/release_execution_checklist_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)
- [report_standard.md](/docs/standards/report_standard.md)
- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)
- [startup_compile_playbook_standard.md](/docs/standards/startup_compile_playbook_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
- [startup_probe_standard.md](/docs/standards/startup_probe_standard.md)
- [status_standard.md](/docs/standards/status_standard.md)
- [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)
- [update_standard.md](/docs/standards/update_standard.md)

Generated reports that describe the current repo state belong in `docs/reports/`.
Current implementation progress and active TODOs belong in `docs/status/`.
