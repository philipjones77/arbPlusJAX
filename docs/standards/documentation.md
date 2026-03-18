Last updated: 2026-03-17T00:00:00Z

# Documentation Standard

This standard defines the intended split between theory, implementation, and practical repository documentation.

It also preserves the repo's core `specs/objects/contracts/implementation` structure:

- `docs/specs/` for semantic definitions and invariants
- `docs/objects/` for named runtime catalogs and object inventories
- `contracts/` for binding runtime and API guarantees
- `docs/implementation/` for how the code is built and organized

## Primary split

- `docs/theory/`: mathematical derivations, conceptual methodology, and explanations of why a method is valid
- `docs/implementation/`: how the code is implemented, structured, wrapped, and organized
- `docs/practical/`: how to actually run, validate, benchmark, and operate the repository based on observed numerical and workflow experience

## Placement rules

- if the reader is asking "why is this mathematically true?", place the document in `docs/theory/`
- if the reader is asking "how is this code path built or wired internally?", place the document in `docs/implementation/`
- if the reader is asking "how should I run this, tune this, compare this, or use this safely in practice?", place the document in `docs/practical/`
- semantic definitions still belong in `docs/specs/`
- binding runtime and API guarantees still belong in `contracts/`

## Writing rule

Do not mix mathematical derivation, code-structure notes, and practical operating advice into one undifferentiated document when a cleaner split is possible. Cross-link between them instead.

## Current repo convention

- keep the `specs/objects/contracts/implementation` structure as the primary backbone for semantics, runtime guarantees, object catalogs, and implementation notes
- most current deep material remains in `docs/implementation/`
- [practical/README.md](/home/phili/projects/arbplusJAX/docs/practical/README.md) is the entry point for practical run/use guidance
- [implementation/README.md](/home/phili/projects/arbplusJAX/docs/implementation/README.md) is the entry point for implementation-facing notes
- [theory/README.md](/home/phili/projects/arbplusJAX/docs/theory/README.md) is the entry point for theory notes
- `docs/practical/` adds a separate layer for operational and numerically informed guidance; it does not by itself reclassify the bulk of `docs/implementation/`
