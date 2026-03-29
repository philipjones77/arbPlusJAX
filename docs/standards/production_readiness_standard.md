Last updated: 2026-03-29T00:00:00Z

# Production Readiness Standard

Status: active

## Purpose

This document defines the repo-wide production-readiness governance model.

The cornerstone of the model is Markdown under governed repository surfaces such
as:

- `docs/standards/`
- `docs/status/`
- `docs/reports/`
- `docs/governance/`
- `contracts/`

Automation, workflows, generated reports, and retained artifacts are
subordinate to those governed Markdown surfaces rather than replacing them.

## Core Rule

Production readiness must be represented in three layers:

1. standards:
   what the repo requires
2. status:
   what is planned, in progress, or done
3. generated reports:
   what currently exists and what is still missing according to measurable repo
   signals

The repo must not rely on scattered workflow YAML alone to represent release,
publishing, security, support, or maturity state.

## Required Subsystems

The production-readiness layer should cover at least:

- release and packaging
- docs publishing
- release governance
- security and supply chain
- operational support
- capability and maturity reporting
- category closeout status for the main function families

## Markdown Authority Rule

Markdown in governed docs surfaces is the source of truth for policy and
completion state.

That means:

- workflows implement the documented policy
- generators summarize the documented policy and current repo state
- retained artifacts provide evidence
- no workflow or script should become the only place where readiness intent is
  defined

## Measurement Rule

The repo should maintain at least one generated production-readiness report that
classifies each required subsystem as:

- `present`
- `partial`
- `missing`

and records the owning Markdown and automation surfaces.

## Required Surfaces

At minimum, the repo should provide:

- standards for each major production-readiness area
- status/TODO files for the unfinished areas
- a generated production-readiness report under `docs/reports/`
- a refresh script that regenerates that report
- a test that verifies the checked-in report is current

## Closeout Rule

A production-readiness area should not be called complete unless:

- its governing standard exists
- its status lane exists
- its current-state report exists
- its main automation or workflow scaffold exists
- the repo records whether the area is fully done or still partial

## arbPlusJAX Specialization

For this repo:

- `docs/` remains the source-of-truth documentation tree for active
  development and governance
- `papers/` remains the separate manuscript lane
- production-readiness reporting belongs under `docs/reports/`
- implementation backlog and closeout state belong under `docs/status/`
