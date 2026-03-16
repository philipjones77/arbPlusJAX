# Documentation Governance

Status: active
Version: v1.0
Date: 2026-03-10

## Base Documents

- `docs/architecture.md`
- `docs/documentation_governance.md`
- `docs/project_overview.md`

## Scope

This repository adopts the smplJAX documentation placement standard as the governing structure for arbPlusJAX.

This document defines:

- the intended repo-root folder layout
- the conceptual role of each `docs/` subfolder
- the authority split between `docs/` and top-level `contracts/`
- where new documents should be placed

It does not define mathematical semantics. Those belong in `docs/specs/`.

## Repository Layout

Preferred top-level structure:

- `docs/`: documentation authority and explanation
- `contracts/`: binding runtime and API guarantees
- `src/`: executable implementation
- `tests/`: conformance, regression
- `examples/`: runnable demos and user-facing templates
- `experiments/`: exploratory work, with each subfolder representing a separate experiment
- `tools/`: repository tooling and maintenance scripts
- `benchmarks/`: benchmark scripts and harness integration
- `outputs/`: canonical output artifact directory
- additional standard folders such as `configs/`, `data/`, `output/`, `stuff/`, and `papers/` may be added when the repository starts using them directly

## Docs Layout

The `docs/` tree is organized as:

- `governance/`
- `notation/`
- `standards/`
- `specs/`
- `objects/`
- `theory/`
- `implementation/`
- `status/`

Root-level `docs/` files are reserved for high-level entry documents such as:

- `architecture.md`
- `documentation_governance.md`
- `project_overview.md`

## Authority Split

Use this order when documents overlap:

1. `docs/specs/`
2. `contracts/`
3. `docs/objects/`
4. `docs/theory/`
5. `docs/implementation/`
6. `docs/status/`

Operational guarantees belong in `contracts/`, not under `docs/`.

## Placement Rules

- semantic definitions and invariants go in `docs/specs/`
- runtime/API obligations go in `contracts/`
- named runtime catalogs go in `docs/objects/`
- derivations and explanations go in `docs/theory/`
- workflows, benchmarks, implementation mapping, and methodology notes go in `docs/implementation/`
- roadmaps, current-state summaries, and active TODOs go in `docs/status/`
- structural/process rules go in `docs/governance/` or this file

## Current Repo Mapping

High-level entry documents:

- `docs/architecture.md`
- `docs/documentation_governance.md`
- `docs/project_overview.md`

Governance/process docs:

- `docs/governance/engineering_policy.md`

Current standards docs:

- `docs/standards/function_naming.md`
- `docs/standards/precision.md`

Current implementation/workflow docs:

- `docs/implementation/build.md`
- `docs/implementation/jax_setup.md`
- `docs/implementation/linux_gpu_colab.md`
- `docs/implementation/run_platform.md`
- `docs/implementation/benchmarks.md`
- `docs/implementation/benchmark_process.md`
- `docs/implementation/modules/`
- `docs/implementation/wrappers/`
- `docs/implementation/external/`

Current status docs:

- `docs/status/todo.md`
- `docs/status/audit.md`
- `docs/status/reports/`

Current repo-root mapping notes:

- `contracts/` now exists as the reserved home for binding runtime/API guarantees
- `experiments/` now exists as the reserved home for exploratory subprojects
- `outputs/` now exists as the canonical governed output root
- `results/` remains in the repo as a legacy compatibility output location until tooling is fully converged
