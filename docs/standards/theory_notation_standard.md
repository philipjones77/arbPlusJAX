Last updated: 2026-03-27T00:00:00Z

# Theory And Notation Standard

Status: active

## Purpose

This document defines how arbPlusJAX should document:

- theory and methodology notes under `docs/theory/`
- notation and symbol conventions under `docs/notation/`

This standard consolidates two related concerns:

- what a methodology note must contain
- how mathematical notation is kept stable across docs, code, and examples

## Theory Placement Rule

Put a document in `docs/theory/` when the main question is:

- what is the mathematical method?
- why does the algorithm work?
- what approximation, enclosure, or continuation idea is being used?
- how should a hardened family be interpreted mathematically?

Theory notes are not the place for:

- binding runtime contracts
- active TODO lists
- generated inventories

## Theory Minimum Content Rule

A theory note should usually include:

- the mathematical object or family being discussed
- the core computational method or reduction
- the important numerical regimes or failure modes
- the relation between the theory and the public runtime surface
- the relation between the theory and diagnostics/metadata when relevant

If a family has multiple materially different methods, the note should say how
those methods are distinguished in the repo.

## Theory-To-Implementation Rule

Theory notes should connect clearly to the hardened repo surface, not remain as
detached derivations.

They should reference, where relevant:

- the owning implementation family
- the exposed public API or wrapper surface
- the diagnostics or metadata terms used in reports/examples
- the current family-level verification ledger when the note is about public
  point/basic surfaces or diagnostics-bearing helper layers

## Notation Placement Rule

Put a document in `docs/notation/` when the main question is:

- what symbol or term do we use for this object?
- how does code naming map to mathematical naming?
- what notation is stable across examples, theory notes, and reports?

`docs/notation/notation.md` is the authoritative bridge between repo terms and
mathematical notation.

## Notation Governance Rule

Notation should be:

- stable across theory, examples, and reports
- explicit about code-name versus math-name differences
- updated when public terminology changes materially
- explicit about point/basic surface terminology, diagnostics payload names, and
  the verification-ledger terms used in reports and notebooks
- explicit about the distinction between evaluation-variable AD and
  family-parameter AD for parameterized families

Do not let notebooks, reports, and theory notes silently drift into different
names for the same public concept.

## Naming Bridge Rule

When code names and mathematical names differ, notation documents should record:

- code-facing term
- mathematical term
- any shape or container convention needed to interpret the object

Examples:

- interval versus box
- `prec_bits` versus precision-in-bits
- mode names such as `point`, `basic`, `rigorous`, `adaptive`
- dense/sparse/block/vblock/matrix-free operator terminology
- point/basic surface notation such as `S_point(f)` and `S_basic(f)`
- diagnostics payload notation such as `D_f`

## Reports And Status Rule

Current theory coverage inventories belong in `docs/reports/`.
Theory backlog and missing methodology coverage belong in `docs/status/`.
Public point/basic theory and notation changes should keep the linked
verification and status artifacts current, especially:

- [point_basic_surface_status.md](/docs/reports/point_basic_surface_status.md)
- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)
