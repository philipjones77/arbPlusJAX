Last updated: 2026-03-29T00:00:00Z

# Production Readiness TODO

Status: `in_progress`

## Scope

This file tracks the cross-cutting production-readiness program that sits above
runtime correctness and below public release claims.

## Done

- governing standards now exist for:
  - release and packaging
  - docs publishing
  - release governance
  - security and supply chain
  - operational support
  - capability and maturity reporting
- top-level support files now exist:
  - `CHANGELOG.md`
  - `SECURITY.md`
  - `SUPPORT.md`
- workflow scaffolds now exist for:
  - build verification
  - publish release
  - docs publish
  - dependency audit

## In Progress

- turn release/publish/docs workflows from scaffold to full implementation
- add production-readiness report generation and keep it current
- tighten support matrix and public stability reporting
- close category-level statuses that are still `in_progress`

## Planned

- provenance/attestation implementation
- generated capability/maturity matrix implementation
- full docs publishing stack implementation
- non-editable install and publish verification expansion
