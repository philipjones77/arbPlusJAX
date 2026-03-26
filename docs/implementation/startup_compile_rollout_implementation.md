Last updated: 2026-03-26T00:00:00Z

# Startup Compile Rollout

Use this when a repo has recurring "slow first run", "JAX recompiles too much", or "every process recompiles everything" complaints.

Primary policy:

- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)

Companion standards:

- [caching_recompilation_standard.md](/docs/standards/caching_recompilation_standard.md)
- [lazy_import_standard.md](/docs/standards/lazy_import_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)

## Rollout Goal

Move the repo from accidental compile behavior to an explicit runtime model with:

- stable-shape public entrypoints
- centralized JIT ownership
- persistent cross-process cache reuse
- intentional warmup
- regression measurement
- fast JAX point-mode public kernels as the default bulk-evaluation path where point mode is a meaningful user surface

## Phase 1. Measure The Current Problem

1. Separate:
   - import time
   - JAX backend initialization time
   - cold compile plus first run
   - warm steady-state execution
   - recompile on new shape or changed static control
2. Identify the top 3 to 10 hot surfaces that users hit first.
3. Record the canonical entrypoint for each surface:
   - CLI
   - notebook helper
   - service endpoint
   - benchmark harness

## Phase 2. Stabilize The Public Path

For each hot surface:

1. Pick one stable-shape strategy:
   - fixed batch size
   - `pad_to`
   - shape buckets
   - prepare/apply plan reuse
2. Make that strategy the default documented fast path.
3. Keep compile-relevant controls explicit:
   - dtype
   - mode
   - precision
   - backend
   - static method selectors
4. Remove or de-emphasize ad hoc "just call a jitted function on any shape" examples.

For point-mode families, also require:

1. A JAX-only point-fast kernel or point-fast service surface.
2. No point-mode dependence on Arb, mpmath, host callbacks, or Python adaptive loops in the hot path.
3. `bind_point_batch_jit(...)` or a family-owned equivalent as the default repeated-call surface.

## Phase 3. Centralize Compile Ownership

1. Identify where `jax.jit` currently lives.
2. Move compile policy toward wrapper, API binder, or runtime boundary layers.
3. Make leaf kernels pure and composable where practical.
4. Document invalidation boundaries clearly:
   - shape
   - padding target
   - dtype
   - precision policy
   - mode or method
   - backend

## Phase 4. Fix Process And Cache Reuse

1. Enable persistent compilation cache in every canonical launcher.
2. Use one repo-owned default cache path unless the environment overrides it.
3. Prefer long-lived processes for repeated workloads.
4. If multi-process execution is unavoidable, verify that child processes share the same cache configuration.

Minimum environment policy:

```bash
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR=/path/to/shared/cache
```

## Phase 5. Add Warmup

1. Create a small warmup routine for the most common kernels.
2. Warm up only the stable-shape path, not every possible configuration.
3. Keep warmup bounded and deterministic.
4. Make benchmark harnesses able to run with and without warmup so cold and warm behavior remain measurable.

## Phase 6. Add Regression Checks

1. Add compile probes for representative families.
2. Track:
   - compile-event count
   - cold-start latency
   - warm latency
   - recompile latency on changed shape
3. Add thresholds in CI or release validation.
4. Treat regressions in startup compile behavior as release-blocking when they affect canonical workflows.

## Cross-repo Review Questions

- What is the canonical stable-shape calling path for each hot family?
- Which layer owns `jax.jit`?
- Which entrypoints enable persistent compilation cache today?
- Which workflows still spawn too many fresh Python processes?
- Which examples are still teaching accidental recompilation patterns?
- Which probes would catch the next regression before users do?

## arbPlusJAX Mapping

Current strengths in this repo:

- compile probes already exist for hot function families
- diagnostics already separate compile and recompile behavior
- benchmark runners already enable persistent compilation cache
- many public APIs already expose `pad_to` and prepared-plan reuse
- the repo already has explicit point-fast JAX standards, plans, reports, and category tests

Recommended next steps in this repo:

1. Make cache-enabled runtime launchers consistent beyond the benchmark layer.
2. Standardize warmup for the highest-traffic function families.
3. Make point-mode docs and APIs teach the point-fast JAX path first, with precise/adaptive layers as fallback or validation engines.
4. Add explicit compile-regression thresholds around the existing probes.

## March 2026 tranche

The latest startup-fix tranche addressed the problem structurally instead of
only trimming matrix-free wrappers:

- introduced a shared [lazy_jit.py](/src/arbplusjax/lazy_jit.py) helper for
  lazy `*_jit` exports
- replaced eager alias blocks in:
  - [arb_core.py](/src/arbplusjax/arb_core.py)
  - [nufft.py](/src/arbplusjax/nufft.py)
  - [dirichlet.py](/src/arbplusjax/dirichlet.py)
  - [acb_dirichlet.py](/src/arbplusjax/acb_dirichlet.py)
  - the earlier Jones matrix-free wrappers
- converted [acb_core.py](/src/arbplusjax/acb_core.py) from eager decorator-time
  JIT wrapper construction to a lazy decorator pattern
- deferred heavy transitive imports in [acb_core.py](/src/arbplusjax/acb_core.py)
  so `hypgeom`, `barnesg`, and Dirichlet helpers are loaded only by the
  specific public functions that need them
- deferred `series_missing_impl` loading in
  [acb_dirichlet.py](/src/arbplusjax/acb_dirichlet.py) behind module
  `__getattr__`

Measured import timings under the repo `jax` interpreter after that tranche:

- `arbplusjax`: about `1.0` to `1.5` seconds
- `arbplusjax.acb_core`: reduced from the earlier multi-dozen-second range to
  about `3.7` seconds
- `arbplusjax.acb_dirichlet`: reduced from double-digit seconds to about
  `0.001` seconds after `acb_core` is already loaded

Regression coverage is owned by
[test_startup_compile_policy.py](/tests/test_startup_compile_policy.py).
