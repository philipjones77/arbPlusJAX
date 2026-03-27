Last updated: 2026-03-26T00:00:00Z

# Update Implementation

This note describes how the repo currently implements
[update_standard.md](/docs/standards/update_standard.md).

## Canonical Commands

- refresh generated subset:
  - `python tools/update_repo_artifacts.py`
- drift check without rewriting:
  - `python tools/check_repo_update_drift.py`

## Current Implementation Model

The current repo uses one umbrella generator path plus targeted generators and
freshness tests.

Primary pieces:

- [check_generated_reports.py](/tools/check_generated_reports.py)
  - existing umbrella refresh path for generated docs/report artifacts
- [update_repo_artifacts.py](/tools/update_repo_artifacts.py)
  - update-policy wrapper around the repo refresh path
- [check_repo_update_drift.py](/tools/check_repo_update_drift.py)
  - non-mutating drift checker for CI and local verification
- [generate_public_metadata_registry.py](/tools/generate_public_metadata_registry.py)
  - runtime-critical metadata registry generator
- [report_status_refresh_inventory.py](/tools/report_status_refresh_inventory.py)
  - generated inventory of report/status refresh ownership

## Update Sequence

When a change affects repo-maintained artifacts, the intended sequence is:

1. refresh generated docs/report artifacts
2. regenerate runtime-critical registries
3. refresh inventory documents that record ownership and refresh paths
4. run non-mutating drift checks

In practice, the current refresh wrapper delegates to:

- `python tools/check_generated_reports.py`

The current drift checker runs the targeted freshness tests without rewriting
artifacts.

## Current Boundaries

- runtime metadata should read from static checked-in registry files, not from
  startup-time implementation imports
- report and status inventories should be generated from explicit tool-owned
  inventories
- docs section indexes should be generated, not manually edited
- startup probe outputs are retained benchmark artifacts and are refreshed by
  the owning probe script rather than the generic repo update wrapper

## When To Use Which Command

Use `python tools/update_repo_artifacts.py` when:

- you changed docs layout or added/removed indexed docs
- you changed public metadata inference or public API inventory
- you changed report generators or their source inventories
- you want to refresh the generated subset before commit/push

Use `python tools/check_repo_update_drift.py` when:

- you want CI-style verification
- you want to confirm no checked-in generated artifact is stale
- you want a non-mutating guard after code review or before commit

## Follow-On Work

The current implementation is intentionally thin and reuses the existing report
generation umbrella. If the update surface grows further, the next likely
expansion is a single machine-readable manifest for all maintained artifact
classes and their owning scripts/tests.
