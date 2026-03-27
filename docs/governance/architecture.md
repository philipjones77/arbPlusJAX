Last updated: 2026-02-25T03:51:38Z

# Architecture

arbPlusJAX is a JAX-first implementation of key Arb functionality with four execution modes:

- point: pure point-evaluation (no intervals, no bounds).
- basic: midpoint-based interval kernels with outward rounding.
- adaptive: sampling/Jacobian-based inflation to widen bounds without recompilation.
- rigorous: ball/interval wrappers that enforce outward rounding and explicit remainder bounds.

## Relation to other approaches

- **C Arb (flint)**: true ball arithmetic with arbitrary precision and rigorous error tracking at every step.
- **arbPlusJAX**: float64/complex128 midpoint evaluation plus outward rounding; interval semantics are post‑hoc and do not change compute precision.
- **SciPy / JAX SciPy**: point-only float64/complex128 with optimized kernels; no interval or enclosure semantics.
- **mpmath**: arbitrary-precision point evaluation; interval enclosures only with `mp.iv`.
- **Mathematica**: high-precision point evaluation; no interval enclosures in our harness.

## Core Layout

- `src/arbplusjax/`: primary implementation modules and wrappers.
- `configs/`: canonical checked-in repo-level configuration and profile
  templates.
- `src/arbplusjax/hypgeom.py`: special functions, series helpers, and interval logic.
- `src/arbplusjax/*_wrappers.py`: mode dispatch (`basic|adaptive|rigorous`) for kernels.
- `src/arbplusjax/ball_wrappers.py`: rigorous ball semantics using Arb-style outward rounding.
- `src/arbplusjax/double_interval.py`: interval arithmetic utilities.
- `src/arbplusjax/acb_*` and `src/arbplusjax/arb_*`: complex/real module families.
- `src/arbplusjax/point_wrappers.py`: point-only kernels (fast path).
- `src/arbplusjax/api.py`: unified entry points for optimized calls.

## Target Internal Package Architecture

The public API should remain stable at the package root, but the internal
runtime layout should converge toward six function-category packages plus
explicit shared helpers.

Target category packages:

- `arbplusjax/core_scalar/`
- `arbplusjax/special/`
- `arbplusjax/dense_matrix/`
- `arbplusjax/sparse_matrix/`
- `arbplusjax/matrix_free/`
- `arbplusjax/transforms/`

Target shared helper layers:

- `arbplusjax/runtime/`
- `arbplusjax/diagnostics/`
- `arbplusjax/validation/`
- `arbplusjax/precision/`
- `arbplusjax/curvature/`
- `arbplusjax/helpers/`

Architectural rules:

- keep the public API stable at the package root
- place category-owned implementations under the six category packages
- move reusable cross-category substrate into explicit helper modules rather
  than hiding it inside one category
- refactor in tranches, not a mass move, so tests/reports/status stay aligned
  after each structural step

### Curvature Layer Placement

Curvature is a cross-cutting helper layer, not a seventh public function
category.

In arbPlusJAX, `curvature` should mean:

- any first-class representation or approximation of local second-order
  structure for objectives, likelihoods, posteriors, operators, or maps
- together with the linear algebra needed to apply, solve, factorize,
  approximate, differentiate, and diagnose that structure

That layer belongs under:

- `arbplusjax/curvature/`

instead of being split ad hoc between dense, sparse, and matrix-free modules.

The curvature layer should stay operator-first and matrix-optional. It should
cover:

- exact Hessians and Hessian-vector products
- generalized Gauss-Newton and Fisher operators
- prior-precision plus likelihood-curvature composition
- posterior-precision operators
- diagonal, block, low-rank, and Lanczos-based approximations
- inverse-diagonal and selected-inverse style marginal extraction
- curvature-aware preconditioners
- implicit-adjoint solve boundaries and transpose-solve policy

The six function categories remain the public runtime decomposition. Curvature
is the governed shared substrate that those categories consume, especially:

- `dense_matrix`
- `sparse_matrix`
- `matrix_free`
- selected `special` and downstream inference workflows

## Optimal Calling Structure (all functions)

Use `arbplusjax.api` for stable, optimized entry points:

- `api.eval_point(name, x)`
  - point-only, no bounds
- `api.eval_point_batch(name, x_batch)`
  - JIT + vmap batched point kernels
- `api.eval_interval(name, x_interval, mode="basic"|"adaptive"|"rigorous", prec_bits, dps)`
  - interval output for a single input
- `api.eval_interval_batch(name, x_interval_batch, mode=..., prec_bits, dps)`
  - JIT + vmap batched interval kernels

These handle both top-level and lower-level functions by name. All interval modes are mode-consistent down the call stack.

Batch JAX kernels are **cached** so each `(function, mode, prec_bits, dps)` combination compiles once per process.

## Recent performance notes

- Added `--jax-warmup` for benchmarks to separate compile time from runtime.
- Bessel basic mode now uses midpoint evaluation; rigorous/adaptive use denser sampling and asymptotic eval for large `z`.

## Execution Modes

Mode dispatch is centralized in `wrappers_common.py` and used by `*_wrappers.py`.

- `point`: `point_wrappers` only; no interval semantics.
- `basic`: midpoint + outward rounding interval kernels.
- `adaptive`: expands bounds using sampling/Jacobian-based estimates.
- `rigorous`: uses ball wrappers or series tail bounds to ensure containment.

## Testing

- `tests/*_chassis.py`: shape, vectorization, and AD-path smoke checks.
- `tests/*_parity.py`: compare against Arb C reference libraries.
- `tests/test_hypgeom_completeness.py`: coverage and helper consistency.

## Results and Benchmarking

- `experiments/benchmarks/outputs/`: benchmark-side generated artifact root.
- `benchmarks/results/`: benchmark logs and sweep run root.
- `configs/`: reviewed benchmark, harness, and runtime configuration profiles
  when the repo uses checked-in config definitions.
- `tools/`: scripts for comparisons and audits.
- `benchmarks/bench_harness.py`: uses `arbplusjax.api` to resolve interval/point modes for consistency.

## Function Registry

See [function_catalog.md](/docs/objects/function_catalog.md) and the generated reports in [reports/README.md](/docs/reports/README.md) for the current public and implementation registries.

## Archived Migration

- `stuff/migration/`: prior migration workspace and C reference builds for parity.
