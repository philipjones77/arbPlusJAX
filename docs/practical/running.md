Last updated: 2026-03-17T00:00:00Z

# Running Guide

This page is for the practical question: how should someone actually run arbPlusJAX in normal use.

## Recommended path

- use [run_platform.md](/home/phili/projects/arbplusJAX/docs/implementation/run_platform.md) as the main workflow reference
- use `tools/run_test_harness.py` for correctness-oriented validation
- use `tools/run_benchmarks.py` for benchmark sweeps and backend comparison
- use [linux_gpu_colab.md](/home/phili/projects/arbplusJAX/docs/implementation/linux_gpu_colab.md) for Colab and Linux GPU runs

## Day-to-day workflow

- chassis validation first
- parity only when Arb C references are available
- benchmark smoke checks separately from correctness
- long benchmark sweeps outside the normal test loop

## Supporting references

- [README.md](/home/phili/projects/arbplusJAX/README.md)
- [tests/README.md](/home/phili/projects/arbplusJAX/tests/README.md)
- [benchmarks/README.md](/home/phili/projects/arbplusJAX/benchmarks/README.md)
- [build.md](/home/phili/projects/arbplusJAX/docs/implementation/build.md)
