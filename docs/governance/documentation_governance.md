# Documentation Governance

Status: active
Version: v1.2
Date: 2026-03-21

## Base Documents

- `docs/governance/architecture.md`
- `docs/governance/documentation_governance.md`
- `docs/project_overview.md`
- `docs/standards/jax_api_runtime_standard.md`

## Scope

This repository adopts the smplJAX documentation placement standard as the governing structure for arbPlusJAX.

The repository should keep the `specs/objects/contracts/implementation` structure as a stable backbone:

- `docs/specs/` for semantic definitions and invariants
- `docs/objects/` for named runtime catalogs and object inventories
- `contracts/` for binding runtime and API guarantees
- `docs/implementation/` for code structure, wrapper layout, and implementation mapping

This document defines:

- the intended repo-root folder layout
- the conceptual role of each `docs/` subfolder
- the authority split between `docs/` and top-level `contracts/`
- where new documents should be placed

Repository-wide reusable standards for public JAX API shape, runtime config,
parameter-change discipline, dtype policy including optional documented
overrides, diagnostics, logging, recompilation discipline, and the rule that
tests/benchmarks/software-comparison layers must not slow normal numerical
evaluation belong under `docs/standards/`. The current canonical document for that topic is
`docs/standards/jax_api_runtime_standard.md`.

It does not define mathematical semantics. Those belong in `docs/specs/`.

The current repo mapping should be maintained as a separate report under `docs/reports/`, not embedded in this governance file.

Landing pages and section indexes should also be generated rather than hand-maintained:

- `README.md`
- `docs/project_overview.md`
- `docs/governance/README.md`
- `docs/implementation/README.md`
- `docs/implementation/modules/README.md`
- `docs/implementation/wrappers/README.md`
- `docs/implementation/external/README.md`
- `docs/standards/README.md`
- `docs/reports/README.md`
- `docs/status/README.md`
- `docs/theory/README.md`
- `docs/reports/current_repo_mapping.md`

## Link Policy

Documentation must not use host-specific absolute filesystem paths such as
`/home/...` or Windows drive roots for cross-references.

Use repository-root links instead:

- use `/docs/...`, `/src/...`, `/tests/...`, `/benchmarks/...`, `/examples/...`, `/contracts/...`, and similar repo-root paths for repository cross-references
- generated docs and reports must emit the same repo-root link style
- machine-local absolute paths are only acceptable in transient command output, not in committed docs, generated reports, or generators

## Repository Layout

Preferred top-level structure:

- `docs/`: documentation authority and explanation
- `contracts/`: binding runtime and API guarantees
- `src/`: executable implementation
- `tests/`: conformance and regression coverage
- `examples/`: runnable demos and user-facing templates
- `experiments/`: exploratory work, with each subfolder representing a separate experiment
- `tools/`: repository tooling and maintenance scripts
- `benchmarks/`: benchmark scripts and harness integration
- `configs/`: checked-in repo-level configuration definitions and profile
  templates
- `benchmarks/results/`: canonical benchmark run artifact directory
- `outputs/`: canonical top-level root whose named subfolders contain retained generated artifacts and other permanent run outputs
- `data/`: local or shared datasets, including large generated inputs that do not belong under source or docs trees
- `experiments/benchmarks/outputs/`: experiment-local benchmark diagnostics and scratch artifact directory
- additional standard folders such as `stuff/` and `papers/` may be added when
  the repository starts using them directly

Artifact storage rule:

- put permanent generated artifacts under named subfolders of `outputs/` unless a narrower canonical subroot already exists
- keep benchmark run trees under `benchmarks/results/`
- keep semi-permanent retained artifacts under named subfolders of `outputs/`
- use `experiments/<name>/outputs/`, `artifacts/`, or `cache/` only for experiment-local material that has not been promoted into the canonical top-level `outputs/` tree
- large data should not be committed casually into normal Git-tracked source trees; if versioned sharing on GitHub is required, use the repository's large-file path such as Git LFS or an external artifact store
- top-level `results/` should not exist; if it appears, it should be treated as a legacy mistake and removed after migrating any needed contents

Tools placement rule:

- `tools/` is for repo utilities, maintenance scripts, report/status generators, packaging helpers, runtime/bootstrap helpers, and correctness-oriented harness entrypoints
- `tools/` is not the home for benchmark implementations, benchmark comparisons, benchmark smoke scripts, or benchmark-specific launchers
- benchmark-facing scripts belong under `benchmarks/`
- example-facing runnable notebooks and scripts belong under `examples/`
- exploratory or retained large-scale runs belong under `experiments/`

## Docs Layout

The `docs/` tree is organized as:

- `governance/`
- `notation/`
- `standards/`
- `specs/`
- `objects/`
- `theory/`
- `implementation/`
- `practical/`
- `reports/`
- `status/`

Large documentation subtrees should have a section `README.md` when they act as
browsable indexes rather than single-document folders. Those README surfaces
should be generated once the subtree becomes large enough that hand-maintained
lists drift.

The `docs/standards/` folder is the canonical home for cross-library and
cross-subsystem standards such as documentation placement, naming conventions,
runtime policy, public API shape, dtype policy, diagnostics policy, and logging
discipline.

High-level governance entry documents should live under `docs/governance/`. The
docs-root landing page for the whole documentation tree is
`docs/project_overview.md`. The current entry documents are:

- `docs/governance/architecture.md`
- `docs/governance/documentation_governance.md`
- `docs/project_overview.md`

## Authority Split

The `specs/objects/contracts/implementation` structure is intentional and should be preserved as the core documentation and guarantee layout of the repository.

Use this order when documents overlap:

1. `docs/specs/`
2. `contracts/`
3. `docs/objects/`
4. `docs/theory/`
5. `docs/implementation/`
6. `docs/practical/`
7. `docs/reports/`
8. `docs/status/`

Operational guarantees belong in `contracts/`, not under `docs/`.

## Placement Rules

- semantic definitions and invariants go in `docs/specs/`
- runtime and API obligations go in `contracts/`
- named runtime catalogs go in `docs/objects/`
- derivations and explanations go in `docs/theory/`
- code-structure notes, wrapper or module layout notes, and implementation mapping go in `docs/implementation/`
- cross-library or cross-subsystem standards for public APIs, runtime config, parameter-change behavior, dtype handling including optional overrides, diagnostics, logging, recompilation policy, and non-intrusive test/benchmark/comparison discipline go in `docs/standards/`
- workflows, runbooks, benchmarking practice, and numerically informed operating guidance go in `docs/practical/`
- function catalogs, function lists, repository inventories, and other report-style reference lists go in `docs/reports/`
- roadmaps, current-state summaries, and active TODOs go in `docs/status/`
- structural and process rules go in `docs/governance/` or this file

## Implementation Naming Rule

Implementation-facing markdown under `docs/implementation/` and its indexed
subtrees should use the suffix:

- `*_implementation.md`

This applies to:

- direct implementation notes in `docs/implementation/`
- module notes in `docs/implementation/modules/`
- wrapper notes in `docs/implementation/wrappers/`
- external lineage reviews in `docs/implementation/external/`

The only exception is generated section indexes named `README.md`.

## Mapping Rule

Current repository mapping should be maintained in a separate report document:

- `docs/reports/current_repo_mapping.md`

That report may list:

- current high-level entry documents
- current standards, theory, implementation, practical, and status documents
- current repo-root mapping notes
- transitional notes about legacy paths

## Commit And Push Rule

Before commit/push, refresh generated docs, reports, notebook inventories, and
status/report indexes through:

- `python tools/check_generated_reports.py`

That command is the canonical repo-side refresh path for generated docs/report
surfaces and the pytest contracts that enforce them.
