Last updated: 2026-03-29T00:00:00Z

# Security And Supply-Chain Standard

Status: active

## Purpose

This document defines the minimum security, dependency, and release-integrity
surfaces for the repo.

## Required Surfaces

The repository should maintain:

- `SECURITY.md`
- dependency-audit workflow
- release provenance/attestation policy
- clear license/notice process for third-party material

## Dependency Rule

Dependencies and optional extras should be auditable and reviewed through a
governed workflow rather than only via local install success.

## Provenance Rule

Release artifacts should support provenance or attestation where the platform
allows it.

## Notice Rule

The repo should make the role of `LICENSE` and `NOTICE` explicit and should be
able to extend third-party notice handling as integrations expand.

## Required Evidence

The repo is compliant only if:

- security entrypoints exist
- dependency-audit scaffolding exists
- provenance is at least planned in release automation
