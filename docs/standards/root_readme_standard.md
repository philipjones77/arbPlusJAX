Last updated: 2026-03-22T00:00:00Z

# Root README Standard

Status: active

## Purpose

This document defines the role of the repo-root `README.md`.

The root README is the top-level landing page for the repository.

It should answer:

- what this repo is
- where the main code lives
- where tests, benchmarks, examples, experiments, standards, reports, and status docs live
- what the primary run surfaces are

## Content Rule

The root README should stay high-level.

It should:

- describe the repo layout
- point to the main run/test/benchmark/example entrypoints
- point to the standards, reports, and status indexes
- prefer short summaries over long implementation detail

It should not:

- duplicate large chunks of subordinate docs
- become the canonical home for detailed runbooks
- become the canonical home for evolving implementation status

## Automation Rule

The root `README.md` should be generated and refreshed automatically.

Its content should be derived from stable repo structure and the generated
`docs/standards/README.md`, `docs/reports/README.md`, and `docs/status/README.md`
surfaces rather than hand-maintained prose.
