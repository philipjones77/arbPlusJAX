Last updated: 2026-03-29T00:00:00Z

# Capability And Maturity Standard

Status: active

## Purpose

This document defines the required repo-wide capability and maturity reporting
surface for public numerical families.

## Required Capability Matrix

The repository should maintain one canonical generated matrix or report that can
answer, at minimum, whether a public surface or family supports:

- `jit`
- `vmap`
- variable AD
- parameter AD
- CPU
- GPU
- diagnostics
- fallback
- maturity level

## Maturity Levels

Preferred maturity values:

- `stable`
- `experimental`
- `benchmark-first`
- `debug-only`

## Required Form

The capability surface should be:

- generated rather than hand-maintained when large
- stable enough for downstream inspection
- documented in one governed location

## Required Evidence

The repo is compliant only if:

- the standard exists
- a status/todo lane exists for implementing it
- the eventual report target has a governed home under `docs/reports/`
