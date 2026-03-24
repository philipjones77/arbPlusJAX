Last updated: 2026-03-17T00:00:00Z

# Running Guide

This page is for the practical question: how should someone actually run arbPlusJAX in normal use.

## Recommended path

- use [run_platform_implementation.md](/docs/implementation/run_platform_implementation.md) as the main workflow reference
- use `tools/run_test_harness.py` for correctness-oriented validation
- use `benchmarks/run_benchmarks.py` for benchmark sweeps and backend comparison
- use [linux_gpu_colab_implementation.md](/docs/implementation/linux_gpu_colab_implementation.md) for Colab and Linux GPU runs
- use [requirements-colab.txt](/requirements-colab.txt) and [colab_bootstrap.sh](/tools/colab_bootstrap.sh) for a CPU-safe Colab bootstrap that matches the local source-tree install

## Day-to-day workflow

- chassis validation first
- parity only when Arb C references are available
- benchmark smoke checks separately from correctness
- long benchmark sweeps outside the normal test loop
- keep Linux, Windows, and Colab on the same source tree and the same repo entrypoints; switch backend mode with environment/harness flags rather than editing notebooks or scripts

## Supporting references

- [README.md](/README.md)
- [tests/README.md](/tests/README.md)
- [benchmarks/README.md](/benchmarks/README.md)
- [build_implementation.md](/docs/implementation/build_implementation.md)
