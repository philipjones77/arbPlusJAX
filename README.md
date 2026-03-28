# arbPlusJAX

arbPlusJAX is an independent JAX implementation derived from Arb and FLINT.
It is not the official Arb project.

## Layout

- `src/arbplusjax/`: primary runtime source tree
- `tests/`: correctness and contract test surface
- `benchmarks/`: benchmark and comparison surface
- `configs/`: checked-in repo-level configuration and profile definitions
- `papers/`: long-form LaTeX manuscript and publication projects
- `examples/`: canonical demonstration notebooks
- `experiments/`: larger exploratory work and retained experiment outputs
- `tools/`: harnesses, generators, and repo utilities
- `docs/`: theory, implementation, practical, standards, reports, and status docs
- `contracts/`: binding runtime and API guarantees

## Primary Run Surfaces

- tests: `python tools/run_test_harness.py --profile chassis --jax-mode cpu`
- benchmarks: `python benchmarks/run_benchmarks.py --profile quick`
- runtime check: `python tools/check_jax_runtime.py --quick-bench`
- examples: see [examples/README.md](/examples/README.md)

Tests, benchmarks, and notebooks are expected to run against the source tree in `src/arbplusjax` by default.

## Documentation Entry Points

- project overview: [docs/project_overview.md](/docs/project_overview.md)
- standards: [docs/standards/README.md](/docs/standards/README.md)
- reports: [docs/reports/README.md](/docs/reports/README.md)
- status: [docs/status/README.md](/docs/status/README.md)
- theory: [docs/theory/README.md](/docs/theory/README.md)
- practical run guidance: [docs/practical/README.md](/docs/practical/README.md)
- implementation notes: [docs/implementation/README.md](/docs/implementation/README.md)

## Install

Editable install:

```bash
python -m pip install -e .
```

Colab bootstrap:

```bash
python -m pip install -r requirements-colab.txt
```

Direct source-tree test run:

```bash
PYTHONPATH=src python -m pytest tests -q -m "not parity"
```

## Notes

- JAX is the primary implementation surface.
- The package root uses lazy public-module loading to keep import-time cost low.
- Reference software and external engines are validation/comparison layers, not the default runtime path.
- Linux, Windows, WSL, GitHub submission, and Colab all use the same source tree; Colab has a CPU-safe bootstrap surface in [requirements-colab.txt](/requirements-colab.txt) and [tools/colab_bootstrap.sh](/tools/colab_bootstrap.sh).
- Optional comparison/reference backends are tracked in [configs/optional_comparison_backends.json](/configs/optional_comparison_backends.json); Mathematica, `c_arb`, `mpmath`, `scipy`, `jax.scipy`, and experimental JAX paths are comparison layers, not mandatory runtime dependencies.
- See [NOTICE](/NOTICE) for acknowledgments and reference links.
