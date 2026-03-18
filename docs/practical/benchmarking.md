Last updated: 2026-03-17T00:00:00Z

# Benchmarking Guide

This page is for the practical question: how should benchmark runs be structured and interpreted in this repository.

## Recommended path

- use [benchmarks.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmarks.md) for the main benchmark workflow
- use [benchmark_process.md](/home/phili/projects/arbplusJAX/docs/implementation/benchmark_process.md) for benchmark process and reporting expectations
- use [testing_harness.md](/home/phili/projects/arbplusJAX/docs/implementation/testing_harness.md) for backend roles and comparison policy

## Practical rules

- treat `tests/` as correctness and `benchmarks/` as performance/comparison
- use Arb / FLINT as the interval reference when available
- use mpmath, Mathematica, and Boost to arbitrate difficult point-value questions
- use SciPy and JAX SciPy only as external comparison backends, not as runtime implementation dependencies
- keep benchmark smoke coverage lightweight; schedule full sweeps separately

## Outputs

- write benchmark artifacts under `outputs/` going forward
- keep `results/` only for compatibility with older runs and tooling
