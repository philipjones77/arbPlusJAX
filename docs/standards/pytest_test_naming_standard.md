Last updated: 2026-03-22T00:00:00Z

# Pytest Test Naming Standard

Status: active

## Purpose

This document defines the naming scheme for files under `tests/`.

The goal is:

- make test ownership obvious from the filename
- keep function-family and module-family tests easy to find
- keep broad correctness, parity, contract, and runtime checks clearly separated
- prevent the test suite from drifting into ad hoc names

## Base Shape

Test files should use:

`test_<family>_<kind>.py`

Where:

- `<family>` identifies the module, function family, or subsystem
- `<kind>` identifies what kind of test it is

Examples:

- `test_arb_core_chassis.py`
- `test_hypgeom_parity.py`
- `test_jrb_mat_logdet_contracts.py`
- `test_sparse_point_api.py`

Legacy/single-owner exceptions are still allowed when a family is currently
represented by one focused file, for example:

- `test_nufft.py`
- `test_incomplete_gamma.py`
- `test_boost_hypgeom.py`

## Family Token Rules

Use one of these family styles:

- module family:
  - `arb_core`
  - `acb_mat`
  - `jrb_mat`
- function family:
  - `hypgeom`
  - `gamma`
  - `bessel`
  - `nufft`
- subsystem family:
  - `dense`
  - `sparse`
  - `matrix_free`
  - `runtime`
  - `api`

Prefer stable mathematical or subsystem names over incidental implementation detail.

## Canonical Kind Tokens

New test files should use one of these canonical `<kind>` tokens whenever possible:

- `chassis`
  Primary implementation/correctness coverage for a module or family.
- `parity`
  External/reference parity coverage.
- `contracts`
  Shape, dtype, routing, return-structure, and API contract checks.
- `modes`
  Point/basic/precision/mode-wrapper coverage.
- `metadata`
  Public metadata and provenance structure checks.
- `diagnostics`
  Diagnostics payload and runtime metadata checks.
- `api`
  Public API/facade entry-point checks.
- `surface`
  Broad public-surface coverage for a subsystem.
- `smoke`
  Fast import/invocation smoke coverage.
- `adjoints`
  Gradient, VJP, JVP, or adjoint-specific checks.
- `special`
  Focused mathematical subfamily tests within a larger family.
- `hardening`
  Stability, containment, and difficult-region regression checks.
- `manifest`
  Runtime/benchmark manifest and environment-capture checks.
- `naming`
  Naming/report/policy structure checks.

## Allowed Transitional Kind Tokens

These tokens exist in the current repo and remain allowed, but new files should
prefer the canonical kinds above unless there is a strong reason not to.

- `adapter`
- `aliases`
- `basic`
- `contract`
- `compat`
- `completeness`
- `complete`
- `engineering`
- `gamma`
- `hypgeom`
- `impls`
- `inventory`
- `inverse`
- `i`
- `k`
- `kernels`
- `layer`
- `mode`
- `new`
- `ops`
- `precision`
- `reports`
- `scaffold`
- `status`
- `tail`
- `tier1`
- `updates`
- `wrappers`

## Naming Guidance By Category

Use these defaults:

- module-owned correctness:
  - `test_<module>_chassis.py`
- module-owned parity:
  - `test_<module>_parity.py`
- public API contract:
  - `test_<family>_contracts.py`
  - `test_<family>_api.py`
- mode wrappers:
  - `test_<family>_modes.py`
- runtime and diagnostics:
  - `test_<family>_diagnostics.py`
  - `test_<family>_manifest.py`
- special-function subfamilies:
  - `test_<family>_special.py`
  - `test_<family>_hardening.py`
- broad subsystem surfaces:
  - `test_<family>_surface.py`
- AD-specific:
  - `test_<family>_adjoints.py`

## What To Avoid

- do not create filenames that give no hint of ownership
- do not encode temporary implementation details into the family token
- do not invent new suffixes when an existing canonical kind already fits
- do not mix parity/reference behavior into ordinary chassis files when a
  separate parity file would be clearer

## Current Repo Policy

- Existing transitional filenames are allowed.
- New test files should prefer canonical kind tokens.
- If a new kind token is truly needed, add it to this standard first.
