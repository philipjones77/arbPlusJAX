Last updated: 2026-03-22T00:00:00Z

# Environment Portability Standard

Status: active

## Purpose

This document defines the portability requirements for running this repo across:

- local Linux
- WSL
- cloud notebook environments such as Google Colab

The goal is to keep examples, tests, benchmarks, and experiments easy to move
between environments without rewriting project structure or runtime logic.

This document owns cross-environment portability rules.

It does not define:

- notebook content requirements
- experiment folder layout
- benchmark grouping or validation semantics

Those belong to:

- [example_notebook_standard.md](/home/phili/projects/arbplusJAX/docs/standards/example_notebook_standard.md)
- [experiment_layout_standard.md](/home/phili/projects/arbplusJAX/docs/standards/experiment_layout_standard.md)
- [benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md)
- [benchmark_validation_policy.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_validation_policy.md)

## Supported Portability Targets

Primary portability targets:

- WSL
- Google Colab

Secondary portability targets:

- native Linux

## Portability Rules

### 1. Environment selection must be explicit

Environment-sensitive flows should use explicit knobs such as:

- `JAX_MODE`
- `JAX_PLATFORMS`
- harness `--jax-mode`

Do not bury platform selection in hard-coded notebook cells or machine-specific
path assumptions.

### 2. Runtime metadata must be capturable

Tests, benchmarks, examples, and experiments should write a shared runtime
manifest when running through the harness or when producing important artifacts.

The manifest should make environment transfer diagnosable by recording at least:

- python path
- OS/platform
- JAX backend/device mode
- key environment settings

### 3. Source tree assumptions must stay repo-relative

Scripts and notebooks should prefer repo-relative paths and bootstrap helpers.

Avoid:

- hard-coded absolute local paths
- machine-specific storage locations
- environment assumptions that only hold inside one workstation setup

### 4. Examples and experiments must separate code from artifacts

To remain transferable:

- scripts/notebooks stay in the example or experiment root
- inputs and outputs live in dedicated subfolders
- runtime artifacts should not be mixed into the source notebook itself unless they are intentionally part of the notebook narrative

### 5. Colab support must use bootstrap, not ad hoc cell drift

Cloud notebook environments should use a maintained bootstrap path when
available.

For this repo, Colab support should prefer:

- [colab_bootstrap.sh](/home/phili/projects/arbplusJAX/tools/colab_bootstrap.sh)

### 6. WSL must be treated as a first-class local platform

WSL is not a special exception path. It is a normal supported run target.

Runbooks, examples, and harness flows should work from WSL without requiring a
separate code layout.

### 7. Benchmark and test harnesses must remain portable

Portability-critical entrypoints should remain:

- `tools/run_test_harness.py`
- `benchmarks/run_benchmarks.py`

These entrypoints should remain the preferred cross-environment launch surfaces.

### 8. Optional integrations must degrade cleanly

When an environment lacks optional software:

- the repo should fall back cleanly
- missing optional systems should be recorded in diagnostics or manifests
- the core workflow should still be runnable where possible

Examples:

- PETSc/SLEPc
- Boost references
- Wolfram local/cloud paths
- GPU availability

## Required Portable Surfaces

The following repo surfaces should remain portable across WSL and Colab:

- example notebooks
- test harness
- benchmark harness
- runtime manifest collection
- experiment layout

## Reports Rule

Current environment support and current portability surfaces belong in
`docs/reports`.
