Last updated: 2026-03-22T00:00:00Z

# Status Standard

Status: active

## Purpose

This document defines what belongs in `docs/status/`.

Status documents describe implementation progress: what is done, what is in
progress, what remains, and what execution plans are active.

Status is not the place for:

- governance policy
- finished inventory reports

## Placement Rule

Put a document in `docs/status/` when the main question is:

- where are we in implementation?
- what remains to be done?
- what is the current completion state?
- what gaps are still open?

Examples:

- TODOs
- completion plans
- gap plans
- active audits
- implementation-stage test coverage plans

## Writing Rule

Status documents should:

- use progress language
- distinguish done / in-progress / planned where relevant
- focus on active implementation state

Status documents should not:

- own policy
- pretend to be final repo inventory
- duplicate reports that belong in `docs/reports/`

## Automation Rule

`docs/status/README.md` should be generated and refreshed automatically.

The generated README should:

- list current status files
- point to key status entrypoints
- avoid stale hand-maintained indexes
- be refreshed by `python tools/check_generated_reports.py`
