Last updated: 2026-03-29T00:00:00Z

# Contracts Surface Standard

Status: active

## Purpose

This document defines what belongs under the repo-root `contracts/` surface and
how contract documents differ from standards, specs, reports, and implementation
notes.

It is intended to remain reusable across engineering-heavy JAX libraries.
arbPlusJAX then specializes it through its current root-level `contracts/`
directory and contract naming.

## Scope

Apply this standard to:

- repo-root `contracts/`
- contract-style runtime/API guarantees
- capability, dtype, mode, matrix, sparse-layout, and stable-subset guarantees

This standard governs placement and document intent. It does not replace the
actual contract content.

## Core Rule

`contracts/` is the home for binding runtime and API guarantees.

If a document answers the question:

- what is guaranteed to downstream callers?

then it belongs in `contracts/`, not in `docs/implementation/`, `docs/status/`,
or `docs/reports/`.

## What Belongs In `contracts/`

Contract documents should define stable obligations such as:

- public API mode guarantees
- dtype and precision guarantees
- capability guarantees
- matrix or sparse layout guarantees
- stable kernel subset guarantees
- selection and routing guarantees

Typical document kinds:

- public API contracts
- capability contracts
- shape/layout contracts
- dtype/precision contracts
- guaranteed subset contracts

## What Does Not Belong In `contracts/`

Do not put these in `contracts/`:

- implementation plans
- roadmap or TODO material
- benchmark inventories
- category status summaries
- exploratory theory notes
- internal refactor notes that do not define downstream guarantees

Those belong respectively in:

- `docs/implementation/`
- `docs/status/`
- `docs/reports/`
- `docs/theory/`

## Writing Rule

Contract documents should:

- state guarantees positively and explicitly
- describe caller-visible obligations and constraints
- avoid implementation-detail drift where possible
- remain stable enough for downstream reference

Contract documents should not:

- narrate development history
- function as a backlog
- blur “current implementation happens to do X” with “the repo guarantees X”

## Naming Rule

Preferred contract naming:

- `*_contract.md`

The root `contracts/README.md` should act as the browsable contract index.

## Relationship To Other Doc Layers

Use this split:

- `docs/specs/`: semantic definitions and invariants
- `contracts/`: binding caller-facing guarantees
- `docs/objects/`: named inventories and catalogs
- `docs/implementation/`: how the code is organized or implemented
- `docs/reports/`: current generated or report-style state
- `docs/status/`: what remains or is in progress

If two documents overlap, the contract wins for caller-visible guarantee
questions.

## Enforceability

The repo is compliant only if:

- binding runtime/API guarantees live in `contracts/`
- contract naming remains explicit and stable
- the contract index exists at `contracts/README.md`
- implementation or status documents do not masquerade as binding guarantees

## arbPlusJAX Specialization

For this repo, `contracts/` is the canonical root-level home for documents such
as:

- public API mode contracts
- capability registry contracts
- dtype and precision contracts
- matrix surface contracts
- sparse layout and operator contracts
- stable kernel subset contracts
