Last updated: 2026-03-04T00:00:00Z

# Benchmark Process

Use this when adding or changing numerical functionality.

## Minimum checks
1. Run tests:
   - `python -m pytest tests -q -m "not parity"`
2. Run benchmark smoke:
   - Windows PowerShell: `$env:ARBPLUSJAX_RUN_BENCHMARKS = "1"`
   - Linux/macOS: `export ARBPLUSJAX_RUN_BENCHMARKS=1`
   - `python -m pytest benchmarks -q -m benchmark`
3. Run quick benchmark sweep:
   - `python tools/run_benchmarks.py --profile quick`
   - (`tools/run_benchmarks.py` uses JAX batched mode by default; add `--no-jax-batch` only when debugging.)

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
python tools/bench_report.py --run <run_dir> --out <run_dir>/report.md
```

Store only curated summaries; raw benchmark artifacts under `experiments/benchmarks/results/` are gitignored by default.
