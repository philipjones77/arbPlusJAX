Last updated: 2026-03-26T00:00:00Z

# Startup Compile Playbook Standard

Status: active

## Purpose

This document is the shared playbook companion to
[startup_compile_standard.md](/docs/standards/startup_compile_standard.md).

It exists for two reasons:

- to make the repo's startup-compile operating model explicit in one place
- to provide a reusable template for sibling repos with the same JAX execution
  model

## Shared Playbook

Every adopting repo should implement the same sequence.

### 1. Define one startup-compile policy

- Treat startup compile as a repo-level runtime policy, not as a benchmark-only
  complaint.
- Require every public JAX hot path to expose a stable-shape execution path.
- Acceptable stable-shape contracts include:
  - fixed batch size
  - `pad_to`
  - shape buckets
  - prepare/apply reuse

### 2. Centralize JIT ownership

- Put `jax.jit` in wrapper, binder, or runtime-boundary layers.
- Do not let every family module invent its own compile policy independently.
- Make the canonical compiled public entrypoint obvious in docs and examples.

### 3. Make fixed-shape calling the default

- The stable-shape path must be the default teaching path.
- `pad_to` should be treated as the public compile contract, not as an obscure
  escape hatch.
- Examples, notebooks, and service snippets should show fixed-shape calling
  first.

### 4. Turn on persistent compilation cache everywhere

- Canonical CLI, service, benchmark, and developer entrypoints must enable:
  - `JAX_ENABLE_COMPILATION_CACHE=1`
  - `JAX_COMPILATION_CACHE_DIR=<shared cache path>`
- Cache configuration should come from one repo-owned helper or bootstrap path,
  not from ad hoc per-script logic.

### 5. Warm up intentionally

- Define a bounded warmup routine for the top hot kernels.
- Warm up the stable-shape path only.
- Do not make the first real user request pay accidental compile cost when the
  workload is known in advance.

### 6. Reuse processes

- Prefer long-lived worker processes for repeated workloads.
- Treat excessive short-lived Python process spawning as a startup-compile
  problem.
- If multi-process execution is unavoidable, shared persistent cache is
  mandatory.

### 7. Add compile budgets to CI

- Maintain representative startup probes.
- Fail CI or release validation when:
  - cold compile latency regresses materially
  - warm latency regresses materially
  - recompilation behavior regresses
  - a representative probe stops compiling

### 8. Separate import, backend init, and compile cost

- Measure at least:
  - import time
  - first backend initialization time
  - first compile plus first real execution
  - steady repeated execution
- Do not collapse these into one undifferentiated startup number.

### 9. Remove dynamic/static argument mistakes

- Audit `static_argnames`, method selectors, precision flags, dtypes, and mode
  switches.
- Compile-relevant controls should be explicit and stable.
- Hidden invalidation boundaries are not acceptable.

### 10. Publish one shared playbook

- Each repo should carry one public startup-compile playbook and one
  implementation template.
- Shared expectations should cover:
  - env vars
  - shape-bucketing rules
  - warmup policy
  - probe structure
  - CI budget policy
  - JIT ownership model

## Required Template Surface

An adopting repo should provide:

- one policy document equivalent to this playbook
- one implementation rollout/template document
- one cache/bootstrap helper or documented canonical environment path
- one startup probe family set with checked-in artifacts
- one test or contract surface proving the stable-shape calling path

## arbPlusJAX Mapping

In this repo, the main corresponding pieces are:

- policy:
  [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- probes:
  [startup_probe_standard.md](/docs/standards/startup_probe_standard.md)
- point default path:
  [point_surface_standard.md](/docs/standards/point_surface_standard.md)
- rollout:
  [startup_compile_rollout_implementation.md](/docs/implementation/startup_compile_rollout_implementation.md)
- cross-repo template:
  [startup_compile_repo_template.md](/docs/implementation/startup_compile_repo_template.md)
