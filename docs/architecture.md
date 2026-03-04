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
- `src/arbplusjax/hypgeom.py`: special functions, series helpers, and interval logic.
- `src/arbplusjax/*_wrappers.py`: mode dispatch (`basic|adaptive|rigorous`) for kernels.
- `src/arbplusjax/ball_wrappers.py`: rigorous ball semantics using Arb-style outward rounding.
- `src/arbplusjax/double_interval.py`: interval arithmetic utilities.
- `src/arbplusjax/acb_*` and `src/arbplusjax/arb_*`: complex/real module families.
- `src/arbplusjax/point_wrappers.py`: point-only kernels (fast path).
- `src/arbplusjax/api.py`: unified entry points for optimized calls.

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

- `results/`: benchmark logs and test runs, including timestamps.
- `tools/`: scripts for comparisons and audits.
- `benchmarks/bench_harness.py`: uses `arbplusjax.api` to resolve interval/point modes for consistency.

## Function Registry

See `docs/references/inventory/function_list.md` for the current public/point/interval function lists.

## Archived Migration

- `stuff/migration/`: prior migration workspace and C reference builds for parity.
