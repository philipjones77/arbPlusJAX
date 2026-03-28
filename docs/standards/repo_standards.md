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
- the role and placement of long-form manuscript projects under `papers/`
- the intended purpose of `AGENTS.md` when present
- the target internal package decomposition beneath the stable package-root API

This is the canonical repo-level communication and placement standard.

It is intentionally more repo-specific than the general JAX runtime standards.
The reusable layer is the communication-surface split itself; the concrete
repo-root files, docs-tree layout, and package decomposition listed here are
arbPlusJAX specializations of that broader pattern.

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
- the target internal package decomposition when the repo is being refactored
  without changing the package-root public API

It belongs in `docs/governance/` because it is a structural governance
document, not an implementation note.

Current architectural direction for the runtime source tree:

- keep the public package API stable at `arbplusjax/__init__.py`
- organize internal runtime code toward six category packages:
  - `core_scalar`
  - `special`
  - `dense_matrix`
  - `sparse_matrix`
  - `matrix_free`
  - `transforms`
- place reusable cross-category substrate in explicit helper layers such as
  `runtime`, `diagnostics`, `validation`, `precision`, `curvature`, or other
  clearly named helper modules
- execute that structural refactor in tranches rather than mass-moving the
  whole runtime tree at once

Curvature-specific rule:

- the repo should treat curvature as a shared helper layer rather than as a
  seventh public runtime category
- operator-first second-order structure such as Hessians, HVPs, GGN, Fisher,
  posterior precision, low-rank/Lanczos approximations, inverse-diagonal
  estimation, and implicit-adjoint curvature solves should converge under
  `arbplusjax/curvature/`
- dense, sparse, and matrix-free modules may expose category-specific public
  surfaces, but the reusable curvature substrate should not be hidden inside
  one matrix family

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
- `papers/`

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

## `papers/` Rule

If the repository maintains publication-grade LaTeX manuscripts, the canonical
home is:

- `papers/`

`papers/` should contain:

- standalone manuscript projects
- paper-specific figures, bibliography files, and build helpers
- long-form publication text that needs a real LaTeX paper structure

`papers/` should not replace the docs tree.

The governing split is:

- `docs/`
  - Markdown-first repo documentation, standards, theory, implementation,
    practical guidance, reports, and status
- `papers/`
  - publication-facing LaTeX manuscripts built from or informed by the
    stabilized docs/runtime surface

This is the intended correct repo structure:

- active documentation and development explanation stays in `docs/`
- paper/manuscript production stays in `papers/`
- `papers/` supplements the docs tree; it does not replace it

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
- `docs/project_overview.md`
- docs section `README.md` files that are generated indexes
- implementation subtree `README.md` files when those folders act as indexed
  document catalogs

See
[generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
for the shared generation policy.
