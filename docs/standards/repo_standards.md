Last updated: 2026-03-23T00:00:00Z

# Repo Standards

Status: active

## Purpose

This document defines the standards for repo-root documents and other top-level
repo-governance surfaces.

It owns:

- the role and placement of the repo-root `README.md`
- the role and placement of `docs/project_overview.md`
- the role and placement of `docs/governance/architecture.md`
- the role and placement of
  `docs/governance/documentation_governance.md`
- what is allowed at the repo root as stable top-level governance material
- the role and placement of repo-level checked-in configuration under `configs/`
- the intended purpose of `AGENTS.md` when present

This is the canonical repo-level communication and placement standard.

## Repo-Root Communication Surfaces

The repo has a small number of top-level communication surfaces with different
audiences.

### Root `README.md`

The root `README.md` is the top-level landing page for the repository.

It should answer:

- what this repo is
- where the main code lives
- where tests, benchmarks, examples, experiments, standards, reports, and
  status docs live
- what the primary run surfaces are

It should stay high-level and should not become the home for detailed runbooks,
implementation status, or large policy prose.

### `docs/project_overview.md`

`docs/project_overview.md` is the repo-structure overview document.

It should explain:

- the stable top-level repo layout
- the docs-tree map
- the relation between runtime code, tests, benchmarks, tooling, contracts,
  and docs

It belongs at `docs/project_overview.md` because it is a docs-tree landing
surface, not a second repo-root README.

### `docs/governance/architecture.md`

`docs/governance/architecture.md` is the high-level architectural map.

It should explain:

- the intended system decomposition
- major subsystem boundaries
- how runtime, contracts, docs, reports, and status fit together

It belongs in `docs/governance/` because it is a structural governance
document, not an implementation note.

### `docs/governance/documentation_governance.md`

`docs/governance/documentation_governance.md` is the placement authority for
the documentation tree.

It should define:

- docs-tree structure
- authority split between `docs/`, `contracts/`, and repo-root surfaces
- placement rules for new documentation
- generated-index and link policy

It belongs in `docs/governance/` because it is the process and placement
authority for repository documentation.

## Repo-Root Allowed Files Rule

The repo root should stay intentionally small.

Allowed stable repo-root governance/communication files include:

- `README.md`
- `LICENSE`
- `NOTICE`
- `AGENTS.md` when present

Allowed stable repo-root structural directories include:

- `src/`
- `tests/`
- `benchmarks/`
- `configs/`
- `examples/`
- `experiments/`
- `tools/`
- `docs/`
- `contracts/`
- `outputs/`
- `data/`

Do not casually add new repo-root prose files when the content actually belongs
under `docs/`.

## `configs/` Rule

If the repository uses checked-in repo-level configuration, the canonical home
is:

- `configs/`

`configs/` should contain reviewed configuration definitions and templates, not
user-local scratch settings.

Each committed config should make its purpose and owning execution surface
clear.

See
[configuration_standard.md](/docs/standards/configuration_standard.md)
for the detailed configuration policy.

## License And Notice Rule

- `LICENSE` is the canonical licensing surface for the repository.
- `NOTICE` is the canonical acknowledgments/reference-notice surface.

They belong at the repo root because they are repository-wide legal and notice
documents rather than ordinary docs-tree content.

## `AGENTS.md` Purpose Rule

If `AGENTS.md` exists at the repo root, its purpose is:

- to describe repository-specific agent/coding instructions
- to capture repo-local execution or editing expectations for coding agents
- to complement, not replace, the human-facing docs tree

It should not:

- duplicate the full README
- become the canonical home for engineering policy already covered by
  standards/governance docs
- become a grab bag of transient TODOs

`AGENTS.md` belongs at the repo root because it is a repo-wide execution aid
for agent tooling rather than a normal user-facing docs page.

## Placement Rule

Use these default locations:

- repo-root landing page: `README.md`
- repo overview map: `docs/project_overview.md`
- repo architecture map: `docs/governance/architecture.md`
- documentation placement/process authority:
  `docs/governance/documentation_governance.md`
- generated docs indexes and section readmes: `docs/`
- repo-wide legal/notice surfaces: repo root

## Automation Rule

The following repo-level landing surfaces should be generated and refreshed
automatically:

- `README.md`
- `docs/index.md`
- `docs/project_overview.md`
- docs section `README.md` files that are generated indexes
- implementation subtree `README.md` files when those folders act as indexed
  document catalogs

See
[generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
for the shared generation policy.
