Last updated: 2026-03-23T00:00:00Z

# Generated Documentation Standard

Status: active

## Purpose

This document defines the policy for generated landing pages and generated
documentation indexes.

This standard consolidates the automation rules that were previously scattered
across:

- `repo_standards.md`
- `report_standard.md`
- `status_standard.md`

Those specialized standards still define audience and content. This document
defines the shared generation rule.

## Generated Surface Rule

The following surfaces should be generated rather than maintained by hand:

- `README.md`
- `docs/index.md`
- `docs/project_overview.md`
- `docs/governance/README.md`
- `docs/implementation/README.md`
- `docs/implementation/modules/README.md`
- `docs/implementation/wrappers/README.md`
- `docs/implementation/external/README.md`
- `docs/standards/README.md`
- `docs/notation/README.md`
- `docs/reports/README.md`
- `docs/status/README.md`
- `docs/theory/README.md`
- `docs/reports/current_repo_mapping.md`

If another docs landing page becomes a repeated source of drift, move it under
the same generator policy.

## Implementation Doc Naming Rule

Implementation-facing markdown under `docs/implementation/` and its indexed
subtrees should use the suffix:

- `*_implementation.md`

Generated section indexes remain `README.md`.

## Generation Source Rule

Generated docs should be derived from:

- stable repo structure
- generated indexes
- canonical standards/governance references
- filesystem-driven folder inventories where section indexes are required

They should not depend on machine-local absolute paths or hand-maintained file
lists.

## Link Rule

Generated docs must use repo-root links such as:

- `/docs/...`
- `/src/...`
- `/tests/...`
- `/benchmarks/...`
- `/examples/...`
- `/contracts/...`

Do not emit host-specific absolute filesystem paths in committed generated
documentation.

## Refresh Rule

Before commit/push, refresh generated docs, reports, inventories, and generated
landing pages through:

- `python tools/check_generated_reports.py`

This is the canonical repo-side refresh path.

## Validation Rule

Generated docs must be protected by pytest ownership.

At minimum:

- generated-doc content should match the current generator output
- generated docs should obey repo link policy

## Relationship To Specialized Standards

- `repo_standards.md` defines the repo-root communication and placement rules
- `report_standard.md` defines what belongs in `docs/reports/`
- `status_standard.md` defines what belongs in `docs/status/`

This document owns the shared rule that those landing pages are generated and
validated as artifacts rather than maintained ad hoc.
