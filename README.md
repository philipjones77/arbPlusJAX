# arbPlusJAX

arbPlusJAX is an independent JAX implementation derived from Arb and FLINT.
It is not the official Arb project.

## Layout

- `src/arbplusjax/`: primary runtime source tree
- `tests/`: correctness and contract test surface
- `benchmarks/`: benchmark and comparison surface
- `examples/`: canonical demonstration notebooks
- `experiments/`: larger exploratory work and retained experiment outputs
- `tools/`: harnesses, generators, and repo utilities
- `docs/`: theory, implementation, practical, standards, reports, and status docs
- `contracts/`: binding runtime and API guarantees

## Primary Run Surfaces

- tests: `python tools/run_test_harness.py --profile chassis --jax-mode cpu`
- benchmarks: `python benchmarks/run_benchmarks.py --profile quick`
- runtime check: `python tools/check_jax_runtime.py --quick-bench`
- examples: see [README.md](examples/README.md)

Tests, benchmarks, and notebooks are expected to run against the source tree in `src/arbplusjax` by default.

## Documentation Entry Points

- standards: [README.md](docs/standards/README.md)
  The standards set is now organized around six concept buckets:
  runtime/numerics, validation/benchmarks/examples, portability/layout,
  contracts/provider boundaries, generated documentation outputs, and
  theory/notation/naming semantics. API calling policy, binder reuse,
  diagnostics, and “keep diagnostics out of the required numeric hot path”
  live under the runtime/contracts side of that model.
- reports: [README.md](docs/reports/README.md)
- status: [README.md](docs/status/README.md)
- practical run guidance: [README.md](docs/practical/README.md)
- implementation notes: [README.md](docs/implementation/README.md)

## Install

Editable install:

```bash
python -m pip install -e .
```

Direct source-tree test run:

```bash
PYTHONPATH=src python -m pytest tests -q -m "not parity"
```

## Notes

- JAX is the primary implementation surface.
- Reference software and external engines are validation/comparison layers, not the default runtime path.
- See [NOTICE](NOTICE) for acknowledgments and reference links.
