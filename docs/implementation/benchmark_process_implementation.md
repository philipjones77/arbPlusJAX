Last updated: 2026-03-04T00:00:00Z

# Benchmark Process

Use this when adding or changing numerical functionality.

Startup compile is not owned by the benchmark process. It is primarily a runtime-entrypoint, launcher, and public-API-shape problem.

Benchmarks only validate whether the runtime policy is working.

Related startup compile guidance:

- [startup_compile_standard.md](/docs/standards/startup_compile_standard.md)
- [startup_compile_rollout_implementation.md](/docs/implementation/startup_compile_rollout_implementation.md)
- [point_fast_jax_standard.md](/docs/standards/point_fast_jax_standard.md)
- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)

## Minimum checks
1. Run tests:
   - `python -m pytest tests -q -m "not parity"`
2. Run benchmark smoke:
   - Windows PowerShell: `$env:ARBPLUSJAX_RUN_BENCHMARKS = "1"`
   - Linux/macOS: `export ARBPLUSJAX_RUN_BENCHMARKS=1`
   - `python -m pytest benchmarks -q -m benchmark`
3. Run quick benchmark sweep:
   - `python benchmarks/run_benchmarks.py --profile quick`
   - (`benchmarks/run_benchmarks.py` uses JAX batched mode by default; add `--no-jax-batch` only when debugging.)

## Baselines to include
- Always: arbPlusJAX interval modes (`basic`, `adaptive`, `rigorous`)
- Prefer when available: C/FLINT Arb (`--c-ref-dir` or auto-detected)
- When appropriate: SciPy and JAX-SciPy
- For high-precision reference: mpmath
- Optional: Mathematica
- Optional: Boost (`--with-boost --boost-ref-cmd "<command>"`)

## Reporting
Generate a markdown summary:

```bash
python benchmarks/bench_report.py --run <run_dir> --out <run_dir>/report.md
```

Store only curated summaries; raw benchmark artifacts under `benchmarks/results/` are gitignored by default.

## Startup Compile Validation

- Benchmark changes must not hide avoidable recompilation churn behind one-off process startup.
- When compile cost is part of the practical calling contract, measure cold, warm, and changed-shape behavior separately.
- Prefer stable-shape benchmark paths such as `pad_to`, fixed batch size, or prepared-plan reuse.
- Canonical benchmark launchers should run with persistent JAX compilation cache enabled.
