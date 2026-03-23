Last updated: 2026-03-23T00:00:00Z

# Configuration Standard

Status: active

## Purpose

This document defines the canonical repo policy for checked-in configuration
files and configuration directories.

It answers:

- where repo-owned configuration should live
- what kinds of configuration belong in version control
- what metadata each committed configuration should carry

## Canonical Location

Repo-level checked-in configuration belongs in:

- `configs/`

Use `configs/` when the configuration is part of the repository contract rather
than a user-local machine preference.

Examples:

- runtime profile templates
- benchmark profile definitions
- harness execution profiles
- repo-specific tool configuration
- documented environment presets that should be reviewed and versioned

Do not put user-local editor preferences, ad hoc scratch files, or ephemeral
machine-specific settings into `configs/`.

## Placement Rule

Use this split:

- `configs/`
  for versioned repo-level configuration definitions
- `docs/`
  for explanatory prose about configuration policy
- `contracts/`
  for binding runtime/API guarantees rather than config storage
- user-local dotfiles or IDE settings
  only for machine-specific preferences that are not part of the repo contract

## Required Configuration Metadata

Each committed configuration should make the following clear, either inside the
file itself or in an adjacent companion note:

- purpose
- owning tool, subsystem, or workflow
- expected execution surface
- required inputs, secrets, or environment variables
- backend scope:
  - CPU-only
  - CPU/GPU portable
  - backend-specific

Recommended additional fields when relevant:

- default dtype or precision expectations
- whether the config is intended for tests, benchmarks, examples, or tooling
- whether the config is stable, experimental, or transitional

## Portability Rule

Checked-in configuration should be portable by default.

- avoid embedding machine-specific absolute filesystem paths
- avoid hard-coding user-specific home directories
- document backend assumptions explicitly
- prefer repo-relative paths where a file path must be stored

## Naming Rule

Configuration filenames should be explicit about their scope.

Prefer names such as:

- `cpu_smoke_profile.json`
- `gpu_benchmark_profile.yaml`
- `matrix_surface_runtime.toml`

Avoid vague names such as:

- `config.json`
- `settings.yaml`

## Repo-Root Rule

`configs/` is an allowed stable repo-root structural directory.

The repo root should not accumulate many one-off configuration files when they
can live under `configs/` with clearer ownership.

## Documentation Rule

The repository should document `configs/` as part of the stable repo layout in:

- `docs/project_overview.md`
- `docs/governance/architecture.md`
- `README.md`

The structural repo policy for `configs/` also belongs in:

- [repo_standards.md](/docs/standards/repo_standards.md)
