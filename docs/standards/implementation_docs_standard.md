Last updated: 2026-03-29T00:00:00Z

# Implementation Docs Standard

Status: active

## Purpose

This document defines what belongs under `docs/implementation/` and how
implementation-facing documentation differs from contracts, standards, reports,
and status files.

It is intended to remain reusable across engineering-heavy JAX libraries.
arbPlusJAX then specializes it through its current `docs/implementation/`
subtrees for modules, wrappers, and external lineage reviews.

This standard does not own code-local docstrings or inline comments inside the
runtime source tree. That is governed by:

- [code_documentation_standard.md](/docs/standards/code_documentation_standard.md)

## Scope

Apply this standard to:

- `docs/implementation/`
- `docs/implementation/modules/`
- `docs/implementation/wrappers/`
- `docs/implementation/external/`

## Core Rule

`docs/implementation/` is the home for code-structure explanation and
implementation mapping.

If a document answers questions like:

- how is this implemented?
- how is this subsystem structured?
- how do modules, wrappers, or helpers relate?
- what external implementation lineage or review informed this code?

then it belongs in `docs/implementation/`.

## What Belongs In `docs/implementation/`

Implementation docs should cover:

- subsystem decomposition
- module ownership and boundaries
- wrapper structure and dispatch layers
- prepare/apply or binder organization
- implementation lineage or review notes
- rollout implementation notes for major refactors or tranches

Typical document kinds:

- module implementation notes
- wrapper implementation notes
- tranche implementation notes
- external review / lineage notes
- build/runtime/toolchain implementation notes

## What Does Not Belong In `docs/implementation/`

Do not use implementation docs for:

- binding caller guarantees
- active TODOs or backlog
- generated inventories
- purely mathematical derivations
- practical user runbooks

Those belong respectively in:

- `contracts/`
- `docs/status/`
- `docs/reports/`
- `docs/theory/`
- `docs/practical/`

## Naming Rule

Implementation-facing Markdown should use:

- `*_implementation.md`

This applies to:

- direct implementation notes in `docs/implementation/`
- module notes in `docs/implementation/modules/`
- wrapper notes in `docs/implementation/wrappers/`
- external lineage reviews in `docs/implementation/external/`

The only exception is generated section indexes named `README.md`.

## Writing Rule

Implementation docs should:

- explain structure and ownership clearly
- map code paths to concepts or layers
- distinguish current implementation from intended direction when needed
- stay concrete enough to help future maintainers navigate the code

Implementation docs should not:

- claim binding guarantees that belong in `contracts/`
- serve as the only place where active backlog is tracked
- duplicate generated reports line-for-line

## Relationship To Other Doc Layers

Use this split:

- `docs/standards/`: policy
- `contracts/`: binding guarantees
- `docs/implementation/`: code structure and implementation mapping
- `docs/practical/`: how to use/run the system
- `docs/reports/`: current measured or generated state
- `docs/status/`: what remains or is in progress

When documents overlap, implementation docs answer “how it is built now,” not
“what is guaranteed forever.”

## Indexing Rule

The implementation subtree should remain browsable through generated indexes:

- `docs/implementation/README.md`
- `docs/implementation/modules/README.md`
- `docs/implementation/wrappers/README.md`
- `docs/implementation/external/README.md`

These indexes should be refreshed by the repo’s standard docs index generator.

## Enforceability

The repo is compliant only if:

- implementation-facing notes live under `docs/implementation/`
- implementation naming follows `*_implementation.md`
- generated implementation indexes remain current
- contracts, reports, and status files are not silently replaced by
  implementation notes

## arbPlusJAX Specialization

For this repo, `docs/implementation/` is the canonical home for:

- module notes such as `*_implementation.md`
- wrapper notes for runtime/mode/dispatch structure
- tranche implementation notes for dense, sparse, matrix-free, startup, and
  benchmark process work
- external lineage reviews that inform implementation decisions
