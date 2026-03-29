Last updated: 2026-03-29T00:00:00Z

# Release And Packaging Standard

Status: active

## Purpose

This document defines the release, distribution, packaging, and installation
verification policy for a production-grade JAX library.

It is intended to remain reusable across similar libraries. arbPlusJAX then
specializes it through its current `pyproject.toml`, GitHub workflows, runtime
profiles, and retained release artifacts.

## Scope

Apply this standard to:

- source distributions
- wheels
- install verification
- release automation
- version support policy
- optional dependency groups

## Required Packaging Surface

The repository must provide:

- one canonical Python packaging surface
- explicit optional dependency groups for major user roles
- install verification beyond editable local development mode

Minimum expected extras:

- `docs`
- `dev`
- `bench`
- `compare` when comparison backends exist
- `release` when the release toolchain needs isolated dependencies

## Required CI Verification

Release readiness requires CI coverage for:

- wheel build
- sdist build
- clean install from built artifacts
- import smoke after install
- at least one non-editable test/install lane

Editable install alone is not sufficient release evidence.

## Publish Workflow Rule

The repository should expose a dedicated publish workflow that is separate from
ordinary test CI.

Minimum required publish phases:

- build artifacts
- verify artifacts
- attach provenance/attestation when supported
- publish only from an explicit release/tag path

## Support Matrix Rule

The repo must maintain one canonical support policy covering:

- supported Python versions
- supported JAX ranges
- supported primary platforms
- support status of optional CPU/GPU environments

This policy must be documented in one governed location rather than inferred
from ad hoc CI files.

## Versioning Rule

Release versioning, deprecation timing, and compatibility expectations must be
documented before publish automation is considered complete.

## Required Evidence

The repo is compliant only if it has:

- a governed packaging standard
- explicit extras in packaging metadata
- a build-verification workflow
- a publish workflow scaffold or implementation
- a support matrix document
