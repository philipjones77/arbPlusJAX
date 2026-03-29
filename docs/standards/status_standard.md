Last updated: 2026-03-22T00:00:00Z

# Status Standard

Status: active

## Purpose

This document defines what belongs in `docs/status/`.

Status documents describe implementation progress: what is done, what is in
progress, what remains, and what execution plans are active.

This document should remain reusable for other engineering-heavy JAX
repositories.

arbPlusJAX then specializes it through its current split between:

- a top-level status index
- dedicated subsystem/program status files
- generated section indexes and status freshness checks

The status layer may include both:

- a top-level TODO index
- dedicated subsystem or program TODO/plan files
- dedicated operational-readiness TODO files for release, docs publishing,
  security, support, and capability reporting when those programs grow beyond a
  single subsection

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

When one broad TODO becomes too mixed, split it into dedicated files by program
or subsystem instead of keeping all backlog detail in a single document.

## Writing Rule

Status documents should:

- use progress language
- distinguish done / in-progress / planned where relevant
- focus on active implementation state
- keep ownership readable by separating unrelated backlogs when the file grows
  too large

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

The generated status README should show both:

- the top-level status entrypoints
- the current dedicated program/subsystem status files
