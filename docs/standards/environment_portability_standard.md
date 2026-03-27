Last updated: 2026-03-25T00:00:00Z

# Environment Portability Standard

Status: active

## Purpose

This document defines the portability requirements for running this repo across:

- GitHub submission runners
- native Windows
- local Linux
- Ubuntu in WSL
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

- [example_notebook_standard.md](/docs/standards/example_notebook_standard.md)
- [experiment_layout_standard.md](/docs/standards/experiment_layout_standard.md)
- [benchmark_grouping_standard.md](/docs/standards/benchmark_grouping_standard.md)
- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)

## Supported Portability Targets

Primary portability targets:

- GitHub submission runners
- native Windows
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

When the requested mode is CPU, shared harness helpers should set an explicitly
CPU-only environment rather than merely relying on fallback behavior. That means
CPU validation runs should suppress accidental CUDA-device probing noise through
the shared environment helper layer.

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

- [requirements-colab.txt](/requirements-colab.txt)
- [colab_bootstrap.sh](/tools/colab_bootstrap.sh)

The default Colab bootstrap should be CPU-safe and source-tree compatible.
GPU enablement may be layered on top as an explicit optional step rather than
being the only supported default.

### 6. GitHub submission must validate the portable source-tree contract

GitHub submission automation should validate the same source-tree portability
contract described in the docs.

At minimum, GitHub submission should cover:

- Windows source-tree install and CPU-first chassis validation
- Ubuntu source-tree install and CPU-first chassis validation
- Colab-compatible bootstrap validation on Ubuntu
- generated docs/report refresh validation

GitHub submission should not require optional licensed or native comparison
software to exist, but it should validate the optional-backend policy and keep
the optional compare stack installable where feasible.

### 7. WSL must be treated as a first-class local platform

WSL is not a special exception path. It is a normal supported run target.

Runbooks, examples, and harness flows should work from WSL without requiring a
separate code layout.

### 8. Native Windows must remain a first-class local platform

Windows should not require a different repo structure, a different package
layout, or notebook-specific source edits.

Portable docs should provide Windows launch examples for:

- editable install
- test harness
- benchmark harness

Use platform-appropriate command syntax in the runbooks, but keep the runtime
entrypoints identical.

### 9. Benchmark and test harnesses must remain portable

Portability-critical entrypoints should remain:

- `tools/run_test_harness.py`
- `benchmarks/run_benchmarks.py`

These entrypoints should remain the preferred cross-environment launch surfaces.

### 10. Linux default interpreter policy must prefer JAX

On native Linux, the default interpreter for repo harnesses and validation
scripts should be the shared `jax` environment when it is available.

Use explicit overrides only when needed:

- `--python ...`
- `ARBPLUSJAX_PYTHON=...`

This keeps local tests, examples, and benchmarks aligned to the same JAX
runtime by default instead of drifting onto an arbitrary ambient interpreter.

### 11. Optional third-party and experimental backends must be explicit

Optional backends should be tracked in checked-in repo config rather than only
in prose.

For this repo, the optional comparison/backend policy should distinguish:

- `c_arb` as the primary interval/enclosure reference when available
- `Mathematica` as an optional high-confidence symbolic/numerical reference
- `mpmath` as the portable high-precision point reference
- `scipy` as the default float64 engineering parity reference
- `jax.scipy` as an optional JAX-side comparison backend
- experimental JAX kernels as optional and explicitly non-canonical

These optional backends must never be required for the default runtime path.

The checked-in config authority for this policy is:

- [optional_comparison_backends.json](/configs/optional_comparison_backends.json)

### 12. Optional integrations must degrade cleanly

When an environment lacks optional software:

- the repo should fall back cleanly
- missing optional systems should be recorded in diagnostics or manifests
- the core workflow should still be runnable where possible

Examples:

- PETSc/SLEPc
- Boost references
- Wolfram local/cloud paths
- Mathematica / WolframKernel
- `jax.scipy`
- GPU availability

## Required Portable Surfaces

The following repo surfaces should remain portable across native Windows, WSL,
Linux, and Colab:

- example notebooks
- test harness
- benchmark harness
- runtime manifest collection
- experiment layout
- a CPU-safe bootstrap/install surface
- a checked-in optional-backend policy surface

## Reports Rule

Current environment support and current portability surfaces belong in
`docs/reports`.
