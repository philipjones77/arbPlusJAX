Last updated: 2026-03-23T00:00:00Z

# Contract And Provider Boundary Standard

Status: active

## Purpose

This document defines:

- what belongs in `contracts/`
- how runtime/API guarantees differ from docs-only explanation
- how public provider-facing capability boundaries should be exposed

This standard consolidates two previously implicit concerns:

- contract placement and authority
- cross-repo provider boundary policy

It does not define mathematical semantics. Those belong in `docs/specs/`.
It does not define current implementation progress. That belongs in
`docs/status/`.

## Authority Split

Use this order when obligations overlap:

1. `docs/specs/`
2. `contracts/`
3. `docs/objects/`
4. `docs/theory/`
5. `docs/implementation/`
6. `docs/practical/`
7. `docs/reports/`
8. `docs/status/`

`contracts/` is the canonical home for binding runtime and API guarantees.

## Contracts Placement Rule

Put material in `contracts/` when the main question is:

- what does the runtime guarantee?
- what inputs, outputs, or invariants are binding?
- what compatibility promise does a public surface make?

Examples:

- public API obligations
- stable payload schema guarantees
- public metadata structure guarantees
- public metadata filtering and serialization guarantees
- capability contracts needed by downstream users

Do not put the following in `contracts/`:

- broad repo policy
- mathematical derivations
- implementation walkthroughs
- current gap tracking

## Provider Boundary Rule

arbPlusJAX is the hardened numeric-kernel repo, not the orchestration layer for
other libraries.

Cross-repo integration should prefer thin downstream adapters over direct
imports of repo-internal layout.

The public provider boundary should therefore be expressed in terms of
capabilities, not files.

Examples of acceptable capability-style surfaces:

- double-gamma evaluation
- fragile-regime promotion
- incomplete-Bessel fallback
- metadata/diagnostics describing method and hardening level

Examples of unacceptable provider coupling:

- downstream code importing arbitrary internal helper modules
- downstream dependence on file layout
- moving broad matrix/operator infrastructure into another library just to
  satisfy one consumer

## Provider-Grade Surface Rule

A family is provider-grade only when it has:

- a stable public entrypoint
- explicit runtime parameterization
- metadata sufficient for downstream routing
- diagnostics sufficient for downstream debugging
- tests, examples, and benchmarks that describe the real calling contract

Provider-grade surfaces should make the following inspectable when relevant:

- supported mode surface
- `float32` / `float64` behavior
- CPU/GPU portability expectations
- AD status
- hardening level
- regime or method tags

Provider-grade metadata should also support deterministic report-facing
serialization and explicit filtering so downstream adapters do not have to
inspect private module layout.

## Thin Adapter Rule

When another repo integrates arbPlusJAX, the adapter layer should live on the
consumer side whenever practical.

That adapter should:

- resolve capability entrypoints from stable public surfaces
- avoid depending on repo-internal module layout
- keep backend/provider choice explicit
- preserve metadata and diagnostics for routing and debugging

## Reports Rule

Current provider-grade families, capability inventories, and metadata exports
belong in `docs/reports/`.

Active provider-boundary hardening work belongs in `docs/status/`.
