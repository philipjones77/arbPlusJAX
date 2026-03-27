Last updated: 2026-03-26T00:00:00Z

# Update Standard

This standard defines how the repository stays "up to date" after code, API,
documentation, benchmark, or report changes.

## Scope

This standard governs:

- checked-in generated reports
- checked-in generated indexes
- checked-in registries and inventories
- checked-in status documents with explicit refresh ownership
- the scripts and tests that verify those artifacts are current

It does not replace feature design specs or implementation notes. Those belong
in `docs/specs/` and `docs/implementation/`.

## Placement

Use the following doc split:

- `docs/standards/`
  - policy and required update rules
- `docs/objects/`
  - named artifact catalogs and registry object descriptions
- `docs/implementation/`
  - concrete update workflow, sequencing, and operator notes
- `tools/`
  - generators, refresh wrappers, and drift checkers
- `tests/`
  - freshness and contract enforcement

The canonical documents for this topic are:

- [update_standard.md](/docs/standards/update_standard.md)
- [update_artifacts.md](/docs/objects/update_artifacts.md)
- [update_implementation.md](/docs/implementation/update_implementation.md)

## Required Artifact Classes

The repo update surface must classify artifacts into explicit classes:

- source-authoritative:
  - hand-maintained source documents or code-owned truth
- generated-authoritative:
  - checked-in artifacts that must be regenerated from a canonical script
- manual-authoritative-with-refresh-path:
  - hand-maintained artifacts that are not generated, but whose refresh method
    and owning trigger must still be documented
- runtime-critical-generated:
  - generated artifacts that runtime code depends on directly

Runtime-critical-generated artifacts require both:

- a canonical generator
- a drift test or equivalent checker

## Canonical Repo Update Paths

The repo must provide both of these entrypoints:

- refresh path:
  - `python tools/update_repo_artifacts.py`
- drift-check path:
  - `python tools/check_repo_update_drift.py`

The refresh path may call lower-level generators. The drift-check path must not
silently rewrite artifacts; it should fail when checked-in artifacts are stale.

## Minimum Required Update Coverage

The update system must cover at least:

- generated docs indexes and section `README.md` files
- checked-in public metadata registries
- checked-in function/report/status inventories
- benchmark/status/report refresh inventories
- example notebook generation where notebooks are managed artifacts
- runtime-critical startup and metadata registries where static artifacts are
  used to keep runtime startup minimal

## Update Methodology

Every maintained artifact must have all of the following defined:

- artifact path
- artifact class
- source of truth
- canonical refresh command
- canonical drift checker
- update trigger

The canonical inventory for those fields belongs in
[update_artifacts.md](/docs/objects/update_artifacts.md).

## CI And Local Enforcement

CI must run a non-mutating drift check against the checked-in artifacts.

At minimum, the drift checker must verify:

- generated docs indexes are current
- generated report/status refresh inventory is current
- static public metadata registry is current
- any repo-maintenance inventories referenced by this standard are current

The refresh path should be used locally before commit/push when touching any
area that affects generated artifacts.

## Relationship To Other Standards

This standard coordinates with:

- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [metadata_registry_standard.md](/docs/standards/metadata_registry_standard.md)
- [report_standard.md](/docs/standards/report_standard.md)
- [status_standard.md](/docs/standards/status_standard.md)
- [startup_probe_standard.md](/docs/standards/startup_probe_standard.md)

Those standards define artifact-specific rules. This document defines the repo
maintenance methodology that ties them together.
