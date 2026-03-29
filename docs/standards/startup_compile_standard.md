Last updated: 2026-03-26T00:00:00Z

# Startup Compile Standard

Status: active

## Purpose

This document defines the repo-level policy for reducing JAX startup compile cost and avoiding avoidable recompilation churn.

It governs runtime entrypoints first. Benchmarks are a measurement surface, not the owner of the problem.

It is intended to be reusable across sibling repos with the same JAX execution model.

This document is a companion to:

- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [lazy_loading_standard.md](/docs/standards/lazy_loading_standard.md)
- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)
- [backend_realized_performance_standard.md](/docs/standards/backend_realized_performance_standard.md)

## Scope

Apply this standard to:

- user-facing CLI entrypoints
- services and long-lived worker processes
- benchmark harnesses
- example notebooks that claim production-style calling behavior
- public JAX batch and bound-call APIs

## Core Policy

- Startup compile behavior is a governed runtime-entrypoint concern, not a benchmark-only concern.
- Repeated startup compile pain must be fixed structurally, not by asking users to tolerate a slow first call.
- For public `point` mode, the preferred structural fix is a fast JAX point path rather than routing repeated traffic through slower precise or adaptive machinery.
- Public execution surfaces must expose a stable-shape path for repeated calls.
- Process layout, cache reuse, warmup behavior, and shape discipline must be designed together.
- Cold-start cost, import cost, backend initialization cost, and steady-state runtime must be measured separately.
- Benchmark harnesses should verify startup behavior, but they must not define the only compliant path.

## Required Design Rules

### 1. Stable-shape execution path

- Every public hot-path JAX surface must support a stable-shape calling mode.
- Acceptable patterns include:
  - fixed batch size
  - `pad_to`
  - bucketed shapes
  - prepare/apply surfaces with compile-stable apply calls
- Arbitrary per-call shape churn must not be the default production path when the workload is naturally repeatable.

### 2. Centralized JIT ownership

- `jax.jit` placement should be concentrated in a small number of wrapper, binder, or runtime-boundary layers.
- Numeric leaf helpers should not each create their own independent compile policy unless there is a measured reason.
- Public docs should make the intended compiled entrypoint explicit.
- For public point-mode families, the default compiled entrypoint should normally be the point-fast JAX service surface rather than a precise-mode wrapper.

This rule is not optional. If compile ownership is scattered across many leaf
modules, the repo no longer has one governed startup-compile policy.

### 3. Stable compile-relevant controls

- Dtype, mode, backend selection, precision policy, and static kwargs should remain stable across repeated calls whenever practical.
- If a control legitimately changes the compiled program, the API should treat that as an explicit invalidation boundary.
- Hidden global switches that silently alter compiled behavior are discouraged.

### 4. Persistent compilation cache

- Every benchmark, CLI, service, and canonical run harness must enable a persistent JAX compilation cache.
- The cache directory must be explicit and reusable across process restarts.
- A repo may choose its own cache location, but the policy must set:
  - `JAX_ENABLE_COMPILATION_CACHE=1`
  - `JAX_COMPILATION_CACHE_DIR=<repo-owned or user-configured path>`
- In practice, this should be centralized in one canonical bootstrap/helper
  path so CLIs, services, benchmarks, and local developer flows do not drift.

### 5. Warmup policy

- Hot production surfaces should define an intentional warmup path.
- Warmup should compile a small, representative kernel set for the stable-shape path.
- Warmup is allowed to be optional in developer workflows, but the default production surface should not rely on accidental first-user compiles.

### 6. Process reuse policy

- Multi-step workflows should prefer long-lived processes when cache reuse materially improves user experience.
- Avoid orchestration that spawns many short-lived Python processes for the same workload unless isolation is required.
- If split-process execution is needed, the persistent compilation cache becomes mandatory rather than optional.

### 7. Measurement and regression policy

- Repos must maintain at least one cold/warm/recompile-sensitive probe for representative hot paths.
- CI or release checks should track:
  - cold compile plus first run latency
  - steady-state latency
  - recompilation on changed shape or changed static controls
  - compile-event count where available
- Startup compile regressions should be treated like runtime regressions.

### 8. Shared playbook and template

- The repo should publish one shared startup-compile playbook and one rollout
  template suitable for sibling repos.
- The shared playbook should cover:
  - stable-shape contract choices
  - JIT ownership model
  - persistent-cache environment policy
  - warmup policy
  - startup-probe structure
  - CI budget policy
  - process-reuse expectations
- A repo should not make each subsystem invent its own startup policy from
  scratch.

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

## Anti-patterns

- JIT compiling directly on arbitrary user-sized arrays in the canonical path
- scattering `jax.jit` throughout many modules without one governing runtime boundary
- spawning a fresh Python process for every small job in a workflow that should reuse compiled executables
- mixing import-time slowness, backend initialization, and first-kernel compilation into one undifferentiated metric
- hiding compile-relevant controls behind mutable globals

## Required Evidence

Repos that adopt this standard should provide:

- at least one compile probe or diagnostics script for a representative hot family
- at least one benchmark or runtime check that uses the persistent compilation cache
- at least one public example or contract test showing the stable-shape calling path
- for point-mode families, at least one public compiled point-batch surface that satisfies the fast-JAX contract where the family is intended for repeated bulk evaluation
- implementation-facing rollout notes that identify the main entrypoints and invalidation boundaries
- one implementation rollout note and one reusable repo-template document for
  sibling repos

## Required Structural Fixes

When startup cost is traced to import-time wrapper creation rather than to the
first real kernel execution, the fix must move that cost out of module import.

Required patterns:

- avoid large module-level `name_jit = jax.jit(...)` export blocks in heavy
  runtime modules
- prefer shared lazy-wrapper helpers for public `*_jit` exports
- avoid eager import of heavy transitive families in core runtime modules when
  only a small subset of public functions actually needs them
- when a module has a large number of decorated JAX entrypoints, prefer a lazy
  decorator or equivalent wrapper pattern over paying the decorator-time cost on
  every import
- prefer module `__getattr__` or equivalent lazy namespace expansion for
  optional fallback inventories rather than importing the whole fallback module
  eagerly

## Adoption Checklist

- Identify the top startup pain surfaces.
- Define the stable-shape public path for each surface.
- Move or consolidate `jax.jit` ownership at the runtime boundary.
- Enable persistent compilation cache in every canonical entrypoint.
- Add intentional warmup for the top hot-path kernels.
- Add cold/warm/recompile probes and CI thresholds.
- Update examples and docs so the stable-shape path is the default teaching path.
- Publish the implementation rollout and repo-template documents.

## Repo Mapping In arbPlusJAX

This repo already contains evidence and partial adoption through:

- compile probes in `benchmarks/*compile_probe.py`
- diagnostics in `src/arbplusjax/jax_diagnostics.py`
- cache-enabled benchmark runners in `benchmarks/run_benchmarks.py` and `benchmarks/run_harness_profile.py`
- stable-shape public surfaces and tests built around `pad_to`
- a repo-wide `point fast JAX` program for point-mode public surfaces

The remaining expectation is to treat these pieces as the default operating model rather than isolated benchmark conveniences.

The implementation-facing companions remain:

- [startup_compile_rollout_implementation.md](/docs/implementation/startup_compile_rollout_implementation.md)
- [startup_compile_repo_template.md](/docs/implementation/startup_compile_repo_template.md)
