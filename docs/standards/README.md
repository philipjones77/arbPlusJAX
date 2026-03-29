Last updated: 2026-03-25T00:00:00Z

# Standards

This section holds repo-wide standards and governance-linked policy documents.

The standards are written with a two-layer reading model:

- general JAX-library rule:
  the part intended to be reusable across JAX-first numerical libraries
- arbPlusJAX specialization:
  the repo-specific adaptation, naming, or ownership choice for this library

The detailed standards still overlap in a few places, so the practical reading model is concept-first rather than filename-first. The current standards set consolidates into eight primary concept groups.

## Reading Rule

Read the standards in this order:

1. reusable owner standard
2. companion standard that narrows one aspect of that owner
3. arbPlusJAX specialization standard for a concrete tranche or family

Do not treat every file here as an equal top-level owner.

The preferred split is:

- reusable owner standards:
  standards that should make sense for a general JAX-first numerical library
- arbPlusJAX specialization standards:
  standards that apply the owner rules to this repo's concrete surface kinds, family tranches, naming, or rollout state

## Consolidated Concept Groups

### 1. Runtime, Numerics, and Production Calling

Primary owner:
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)

Specialized companion documents:
- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- [engineering_standard.md](/docs/standards/engineering_standard.md)
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
- [error_handling_standard.md](/docs/standards/error_handling_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)
- [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [logging_standard.md](/docs/standards/logging_standard.md)
- [metadata_registry_standard.md](/docs/standards/metadata_registry_standard.md)
- [precision_standard.md](/docs/standards/precision_standard.md)
- [configuration_standard.md](/docs/standards/configuration_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)

Consolidation note:
- treat `jax_api_runtime_standard.md` as the canonical runtime/API contract
- treat `api_surface_kinds_standard.md` as the canonical taxonomy for direct, light-wrapper, bound-service, compiled-bound, diagnostics-bearing, prepared-plan, and policy-helper public surfaces
- treat `engineering_standard.md` as the hardening and status-interpretation overlay
- treat `backend_realized_performance_standard.md` as the practical backend-performance companion layered on top of structural fast-JAX/runtime readiness
- treat `error_handling_standard.md` as the canonical hard-failure, numerical-status, fallback, and shared status-code companion
- treat `caching_recompilation_standard.md` as the explicit cache, binder-reuse, prepared-plan, and recompilation-discipline companion
- treat `implicit_adjoint_operator_solve_standard.md` as the operator-first solve, transpose-solve, and implicit-adjoint differentiation companion
- treat `lazy_loading_standard.md` as the canonical import-time load and public lazy-boundary companion
- treat `logging_standard.md` as the canonical structured logging, verbosity, and off-hot-path emission companion
- treat `metadata_registry_standard.md` as the canonical public metadata/capability reporting companion
- treat `configuration_standard.md` as the checked-in runtime/optional-backend configuration companion
- treat `core_scalar_service_calling_standard.md` as a tranche-specific specialization, not a second general runtime policy
- API calling shape, binder reuse, diagnostics payloads, error policy, logging hooks, optional backend declaration, fallback visibility, and the rule that diagnostics/profiling stay outside the mandatory numeric hot path all belong to this runtime concept
- `fast_jax_standard.md`, `operational_jax_standard.md`, and `core_scalar_service_calling_standard.md` together replace the older point-specific owner standards

### 2. Validation, Benchmarking, and Executable Examples

Primary owners:
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)

Specialized companion documents:
- [benchmark_grouping_standard.md](/docs/standards/benchmark_grouping_standard.md)
- [api_usability_standard.md](/docs/standards/api_usability_standard.md)
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)

Consolidation note:
- `benchmark_validation_policy_standard.md` owns measurement and benchmark-contract policy
- `benchmark_grouping_standard.md` is the taxonomy companion for the same benchmark concept
- `example_notebook_standard.md` is the executable-teaching analogue of the same validation/communication layer
- `api_usability_standard.md` owns the intended public calling pattern and how notebooks/practical docs should teach it
- `api_usability_standard.md` is broadly reusable; notebook-family or tranche-specific usage documents should be treated as specializations

### 3. Portability, Startup, and Run Layout

Primary owners:
- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)
- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
- [startup_probe_standard.md](/docs/standards/startup_probe_standard.md)

Consolidation note:
- these documents jointly own where things run, how startup/import/compile behavior is governed, and where artifacts live
- GitHub submission, Windows, Linux/WSL, and Colab portability expectations belong here rather than in ad hoc runbook notes
- the startup standards remain separate because they govern different evidence surfaces: compile budget, import boundary, and startup probes
- repo-root path conventions, retained artifact layout, and checked-in notebook policy are arbPlusJAX specializations of the broader portability/layout layer

### 4. Contracts And Provider Boundary

Primary owners:
- [contract_and_provider_boundary_standard.md](/docs/standards/contract_and_provider_boundary_standard.md)
- [contracts_surface_standard.md](/docs/standards/contracts_surface_standard.md)

Consolidation note:
- this document consolidates the missing contract-placement and provider-boundary policy into one public-surface concept
- `contracts_surface_standard.md` owns what belongs in the root `contracts/` layer and how binding guarantees differ from implementation notes or status docs
- downstream-facing API capability contracts, metadata guarantees, and provider-grade surface rules belong here rather than in ad hoc per-family notes
- the generic rule is that libraries should expose stable capability contracts; the arbPlusJAX specialization is which provider and capability terms this repo actually uses

### 5. Documentation, Implementation, and Generated Communication Surfaces

Primary owners:
- [code_documentation_standard.md](/docs/standards/code_documentation_standard.md)
- [specs_standard.md](/docs/standards/specs_standard.md)
- [objects_standard.md](/docs/standards/objects_standard.md)
- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [implementation_docs_standard.md](/docs/standards/implementation_docs_standard.md)
- [report_standard.md](/docs/standards/report_standard.md)
- [status_standard.md](/docs/standards/status_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)
- [update_standard.md](/docs/standards/update_standard.md)

Consolidation note:
- `code_documentation_standard.md` owns code-local docstrings, inline comments, and code-to-doc linkage
- `specs_standard.md` owns what belongs in `docs/specs/` and how semantic specifications differ from contracts, theory, and status
- `objects_standard.md` owns what belongs in `docs/objects/` and how catalogs/inventories differ from generated reports
- `generated_documentation_standard.md` owns the shared generation rule
- `implementation_docs_standard.md` owns what belongs in `docs/implementation/` and how implementation mapping differs from contracts, reports, and status
- `repo_standards.md` owns repo-root communication and placement
- `update_standard.md` is the operational companion for repo-maintained artifact refresh/update discipline
- the report/status standards remain specialized audience documents under the same documentation-output concept
- repo communication and generated landing pages are mostly arbPlusJAX-specific, but the split between governance, implementation, reports, status, and generated indexes should remain reusable

### 6. Release, Security, Support, and Capability Governance

Primary owners:
- [release_packaging_standard.md](/docs/standards/release_packaging_standard.md)
- [docs_build_standard.md](/docs/standards/docs_build_standard.md)
- [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md)
- [release_governance_standard.md](/docs/standards/release_governance_standard.md)
- [security_supply_chain_standard.md](/docs/standards/security_supply_chain_standard.md)
- [operational_support_standard.md](/docs/standards/operational_support_standard.md)
- [capability_maturity_standard.md](/docs/standards/capability_maturity_standard.md)
- [production_readiness_standard.md](/docs/standards/production_readiness_standard.md)

Consolidation note:
- these documents own the missing production-library layer that sits above runtime correctness and beneath public release claims
- `docs_build_standard.md` owns the Markdown-first source/build/rewrite model for Sphinx/MyST publication
- `production_readiness_standard.md` owns the Markdown-first governance model that ties standards, status, reports, and automation together
- release build verification, docs publishing, changelog/support/security entrypoints, and capability reporting should be governed explicitly rather than left as repo folklore

### 7. Theory, Notation, and Naming Semantics

Primary owners:
- [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)
- [function_naming_standard.md](/docs/standards/function_naming_standard.md)

Specialized companion documents:
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)

Consolidation note:
- `theory_notation_standard.md` owns methodology-note and notation governance
- function naming and test naming remain the explicit naming-policy layer
- mathematical-family naming and point/basic terminology are arbPlusJAX specializations layered on top of more general notation discipline

### 8. ArbPlusJAX Specialization Standards

These standards should remain separate from the more reusable owner standards, but they are best read as one specialization layer rather than as many unrelated top-level policies.

Primary specialization documents:
- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)

Consolidation note:
- these documents are intentionally not merged into the general owner standards because they encode repo-specific tranche rules and family-level calling guidance
- read them only after the owner standards for runtime, validation, portability/startup, and release governance

## Detailed Standards
- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- [api_usability_standard.md](/docs/standards/api_usability_standard.md)
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)
- [benchmark_grouping_standard.md](/docs/standards/benchmark_grouping_standard.md)
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [capability_maturity_standard.md](/docs/standards/capability_maturity_standard.md)
- [code_documentation_standard.md](/docs/standards/code_documentation_standard.md)
- [configuration_standard.md](/docs/standards/configuration_standard.md)
- [contract_and_provider_boundary_standard.md](/docs/standards/contract_and_provider_boundary_standard.md)
- [contracts_surface_standard.md](/docs/standards/contracts_surface_standard.md)
- [core_scalar_service_calling_standard.md](/docs/standards/core_scalar_service_calling_standard.md)
- [docs_build_standard.md](/docs/standards/docs_build_standard.md)
- [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md)
- [engineering_standard.md](/docs/standards/engineering_standard.md)
- [environment_portability_standard.md](/docs/standards/environment_portability_standard.md)
- [error_handling_standard.md](/docs/standards/error_handling_standard.md)
- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)
- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)
- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- [function_naming_standard.md](/docs/standards/function_naming_standard.md)
- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [implementation_docs_standard.md](/docs/standards/implementation_docs_standard.md)
- [implicit_adjoint_operator_solve_standard.md](/docs/standards/implicit_adjoint_operator_solve_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [jax_surface_policy_standard.md](/docs/standards/jax_surface_policy_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [logging_standard.md](/docs/standards/logging_standard.md)
- [metadata_registry_standard.md](/docs/standards/metadata_registry_standard.md)
- [objects_standard.md](/docs/standards/objects_standard.md)
- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)
- [operational_support_standard.md](/docs/standards/operational_support_standard.md)
- [precision_standard.md](/docs/standards/precision_standard.md)
- [production_readiness_standard.md](/docs/standards/production_readiness_standard.md)
- [pytest_test_naming_standard.md](/docs/standards/pytest_test_naming_standard.md)
- [release_governance_standard.md](/docs/standards/release_governance_standard.md)
- [release_packaging_standard.md](/docs/standards/release_packaging_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)
- [report_standard.md](/docs/standards/report_standard.md)
- [security_supply_chain_standard.md](/docs/standards/security_supply_chain_standard.md)
- [special_function_ad_standard.md](/docs/standards/special_function_ad_standard.md)
- [specs_standard.md](/docs/standards/specs_standard.md)
- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_import_boundary_standard.md](/docs/standards/startup_import_boundary_standard.md)
- [startup_probe_standard.md](/docs/standards/startup_probe_standard.md)
- [status_standard.md](/docs/standards/status_standard.md)
- [theory_notation_standard.md](/docs/standards/theory_notation_standard.md)
- [update_standard.md](/docs/standards/update_standard.md)

```{toctree}
:maxdepth: 2
:hidden:

api_surface_kinds_standard
api_usability_standard
backend_realized_performance_standard
benchmark_grouping_standard
benchmark_validation_policy_standard
caching_recompilation_standard
capability_maturity_standard
code_documentation_standard
configuration_standard
contract_and_provider_boundary_standard
contracts_surface_standard
core_scalar_service_calling_standard
docs_build_standard
docs_publishing_standard
engineering_standard
environment_portability_standard
error_handling_standard
example_notebook_standard
experiment_layout_standard
fast_jax_standard
function_naming_standard
generated_documentation_standard
implementation_docs_standard
implicit_adjoint_operator_solve_standard
jax_api_runtime_standard
jax_surface_policy_standard
lazy_loading_standard
logging_standard
metadata_registry_standard
objects_standard
operational_jax_standard
operational_support_standard
precision_standard
production_readiness_standard
pytest_test_naming_standard
release_governance_standard
release_packaging_standard
repo_standards
report_standard
security_supply_chain_standard
special_function_ad_standard
specs_standard
startup_compile_standard
startup_import_boundary_standard
startup_probe_standard
status_standard
theory_notation_standard
update_standard
```

Generated reports that describe the current repo state belong in `docs/reports/`.
Current implementation progress and active TODOs belong in `docs/status/`.
