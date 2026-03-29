Last updated: 2026-03-29T00:00:00Z

# Code Documentation Standard

Status: active

## Purpose

This document defines the required code-adjacent documentation standard for the
repository:

- module docstrings
- class and public function docstrings
- inline comments
- code-to-theory and code-to-practical linkage
- short usage examples embedded in code documentation where justified

It is intended to be reusable across engineering-heavy JAX libraries. arbPlusJAX
specializes it through its current mode-driven API, theory tree, example
notebooks, and implementation docs.

## Scope

Apply this standard to:

- `src/`
- `tools/`
- repo-owned executable benchmark and example support code when hand-written
- public runtime modules, helper modules, and maintained support utilities

It does not replace:

- `docs/theory/` for derivations and methodology
- `docs/practical/` for runbooks and production calling patterns
- `docs/implementation/` for module-structure explanation
- `contracts/` for binding guarantees

## Core Rule

Code should be understandable from the source without forcing maintainers to
reconstruct intent from tests or git history.

That means:

- public code surfaces need stable code-local explanation
- non-obvious internal logic needs concise rationale
- mathematical or numerical choices need a documented name, regime, or pointer
- long-form explanation belongs in docs, with code docstrings pointing there

## Required Documentation Surfaces

### Module Docstrings

A module-level docstring is required when the module is one of:

- a public family owner module
- a public helper/runtime substrate module
- a tool or harness entrypoint with repo-facing operational importance

The module docstring should cover:

- what the module owns
- its main surface kinds
- important constraints or boundaries
- where deeper theory or practical docs live when relevant

### Public Class And Payload Docstrings

Public classes, dataclasses, named tuples, and other exported structural types
should have a docstring when they are part of:

- the public API
- diagnostics/status payloads
- prepared-plan payloads
- policy/configuration payloads

The docstring should state:

- what the object represents
- what the important fields mean
- whether it is a stable public payload or an internal structure

### Public Function Docstrings

Hand-written public functions should have a docstring unless they are:

- trivial aliases generated from a shared wrapper factory, and
- already covered by module/family-level documentation plus generated surface
  docs

For non-trivial public functions, the docstring should cover:

- a one-line summary
- the semantic role of the function
- the important arguments when not obvious from naming alone
- what is returned
- important execution notes when relevant:
  - mode behavior
  - dtype/precision behavior
  - CPU/GPU/backend or JAX-transform behavior
  - diagnostics/fallback/error behavior

### Internal Helper Docstrings

Internal helpers do not need blanket docstrings.

They should have a docstring when they are:

- algorithmically non-obvious
- reused across multiple modules
- implementing a subtle invariant or shape/dtype contract
- likely to be touched during future refactors

Do not write boilerplate docstrings for tiny obvious helpers.

## Required Content Rules

### Summary Line

The first line should be a short imperative or descriptive summary of what the
surface does.

### Parameters And Returns

Use concise parameter/return explanation when at least one of these is true:

- the argument meaning is not obvious from the name
- the shape contract matters
- the mode/backend/fallback policy depends on an argument
- the function has more than two materially important parameters

Do not mechanically document obvious scalars that are already self-explanatory.

### Shape / Dtype / Mode Notes

Docstrings should explicitly mention shape, dtype, or mode behavior when any of
those are part of the semantic contract.

Typical cases:

- batched vs unbatched RHS
- `point/basic/adaptive/rigorous`
- real vs complex
- dense vs sparse vs matrix-free
- padded or bucketed batch behavior

### Numerical / Mathematical Notes

When a function implements a named method, regime switch, or approximation, the
docstring should say so briefly.

If deeper explanation is needed:

- name the method in the docstring
- point to the relevant theory/practical/implementation document

Do not embed long derivations in code comments or docstrings.

### Diagnostics / Fallback / Error Notes

If a public surface exposes diagnostics, fallback, or error policy behavior,
the docstring should say that explicitly.

Examples:

- whether diagnostics are returned or available via a companion surface
- whether backend fallback may occur
- whether the function may raise on hard contract failure

## Inline Comment Rule

Inline comments are for:

- why
- invariants
- tricky shape/layout assumptions
- numerical guardrails
- surprising implementation choices

Inline comments are not for restating obvious code.

## Comment Placement Rule

Prefer:

- a short comment before a non-obvious block

Avoid:

- long trailing comments on every line
- dense blocks of narrative prose inside the code body

If a block needs more than a few lines of explanation, move the long-form
explanation into:

- `docs/implementation/`
- `docs/theory/`
- `docs/practical/`

and leave a short pointer in code.

## Example Rule

Docstrings may include a very short example only when it materially improves
correct usage of a public surface.

Long examples belong in:

- canonical notebooks under `examples/`
- practical docs under `docs/practical/`

The code docstring should not become the primary tutorial surface.

## Code-To-Docs Linkage Rule

When a public surface is mathematically or operationally important, code-local
documentation should point outward to the owning docs layer instead of trying to
carry the entire explanation itself.

Preferred mapping:

- derivation or methodology:
  `docs/theory/`
- caller guidance or backend policy:
  `docs/practical/`
- module structure:
  `docs/implementation/`
- binding guarantees:
  `contracts/`

## Generated Wrapper Exception

The repo has many generated or factory-produced public wrappers.

For those surfaces, compliance may be satisfied by:

- a documented family/module owner
- generated API/report surfaces
- canonical notebook/practical coverage

This exception is valid only when the wrapper is truly repetitive and the
meaning is inherited mechanically from a well-documented owner surface.

## Formula And Theory Rule

Short formulas or method names in docstrings are allowed.

Full derivations should not live in code.

If a public surface depends on a non-obvious mathematical contract, the code
documentation should identify one of:

- the named method
- the main invariant
- the relevant theory document

## Enforceability

The repo is compliant only if:

- new or materially changed public hand-written APIs include appropriate
  docstrings
- public diagnostics/policy/plan payloads are documented
- non-obvious numerical blocks are explained with concise comments
- code comments explain why/invariants rather than restating syntax
- long-form theory/practical content is not silently trapped only in code

## arbPlusJAX Specialization

For this repo, code documentation should explicitly account for:

- `point/basic/adaptive/rigorous` mode behavior when relevant
- JAX transform behavior (`jit`, `vmap`, AD) when relevant
- backend-realized performance guidance when relevant
- diagnostics/error/fallback hooks when public
- the split between sparse, dense, matrix-free, and special-function families

This repo should prefer:

- concise code-local docstrings
- deeper explanation in `docs/`
- executable teaching in canonical notebooks

