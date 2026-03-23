Last updated: 2026-03-22T00:00:00Z

# Report Standard

Status: active

## Purpose

This document defines what belongs in `docs/reports/`.

Reports describe the current repo inventory: what files, functions, surfaces,
or generated summaries exist in the repo as it stands.

Reports are not the place for:

- policy or governance
- implementation TODOs
- active roadmap tracking

## Placement Rule

Put a document in `docs/reports/` when the main question is:

- what currently exists?
- what is the current inventory?
- what grouped surfaces are present?
- what generated summary or registry is available?

Examples:

- function inventories
- benchmark group inventories
- notebook inventories
- environment support inventories
- generated capability registries

## Writing Rule

Reports should:

- describe the current repo state
- prefer factual inventory language over roadmap language
- point to the governing standards when policy is relevant

Reports should not:

- own policy
- duplicate standards text
- act as TODO lists

## Automation Rule

`docs/reports/README.md` and repo-mapping reports such as
`docs/reports/current_repo_mapping.md` should be generated and refreshed
automatically.

The generated README should:

- list current report files
- point to the relevant standards when needed
- avoid hand-maintained drift
- be refreshed by `python tools/check_generated_reports.py`
