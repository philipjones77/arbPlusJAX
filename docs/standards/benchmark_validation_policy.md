# Benchmark Validation Policy

Status: active
Version: v1.0
Date: 2026-03-22

## Purpose

This document defines how benchmark-facing pytest coverage and benchmark CLI
entrypoints should be organized in this repo.

This document owns:

- what benchmark concerns mean
- what benchmark outputs should contain
- how official benchmarks are chosen

It does not own the allowed grouping categories or filename-to-group mapping.
Those belong to [benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md)
and [taxonomy.py](/home/phili/projects/arbplusJAX/benchmarks/taxonomy.py).

The goal is to keep benchmark concerns separate:

- implementation correctness and accuracy
- runtime speed and bottleneck detection
- compile and recompile behavior
- AD/forward-backward cost
- backend/reference comparison

## Benchmark Layers

### Accuracy

Use this layer when the benchmark exists to measure error or residual quality.

Typical outputs:

- absolute error
- relative error
- residual norm
- gradient residual
- accuracy against dense truth, Arb truth, SciPy truth, or exact diagonal cases

Pytest marker:

- `benchmark_accuracy`

### Performance

Use this layer when the benchmark exists to measure runtime behavior.

Required split for JAX-heavy paths:

- cold first call
- warm cached execution
- recompile on changed shape or static argument
- Python orchestration overhead where that overhead is material

Pytest marker:

- `benchmark_perf`

### Compile

Use this layer when the benchmark exists primarily to measure:

- cold JIT compilation
- recompile cost on changed shape
- recompile cost on changed compile-relevant static controls

Typical outputs:

- cold compile timing
- recompile timing
- compile/warm ratio
- changed-shape or changed-static-arg sensitivity

Pytest marker:

- `benchmark_compile`

### AD

Use this layer when the benchmark exists primarily to measure:

- forward and backward runtime together
- gradient/JVP/VJP throughput
- gradient residual against a known reference when relevant

Typical outputs:

- forward timing
- backward timing
- full value-plus-gradient timing
- gradient residual or dot-product residual where claimed

Pytest marker:

- `benchmark_ad`

### Comparison

Use this layer when the benchmark exists to compare implementations or external
engines.

Examples:

- JAX-native vs SciPy
- JAX-native vs PETSc/SLEPc
- JAX-native vs optional NUFFT backends

Pytest marker:

- `benchmark_compare`

### Official

Some benchmark scripts are designated as the official benchmark for a benchmark
concern. Official means:

- it is the canonical pytest/harness target for that concern
- dashboards and regression discussions should point to it first
- if another script overlaps, the official script wins unless governance changes

Pytest marker:

- `benchmark_official`

## Repo Implementation Rule

The repo benchmark taxonomy is defined in [taxonomy.py](/home/phili/projects/arbplusJAX/benchmarks/taxonomy.py).

That taxonomy:

- discovers benchmark and comparison CLI scripts
- classifies each one by intent and category
- derives pytest markers for benchmark smoke coverage

The allowed grouping axes and benchmark category families are defined in
[benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md).

New benchmark entrypoints should not be added without classification in the
taxonomy.

## Pytest Usage

Examples:

```bash
pytest -m benchmark
pytest -m benchmark_perf
pytest -m benchmark_compile
pytest -m benchmark_ad
pytest -m benchmark_compare
pytest -m benchmark_official
pytest -m "benchmark_matrix_free and benchmark_perf"
pytest -m "benchmark_transform and benchmark_gpu"
```

## Official Benchmark Selection Rule

Choose one official benchmark for a concern only when it satisfies all of:

- representative:
  - covers the canonical runtime path for that concern
- stable:
  - inputs and reporting shape are stable enough for regression use
- diagnosable:
  - separates the measurements that matter for that concern
- reproducible:
  - can run repeatedly under the harness with controlled environment capture
- comparable:
  - produces artifacts or summaries that make regression comparisons meaningful

Do not designate multiple official benchmarks for the same narrow concern unless
the concern is explicitly split by device or subsystem.

Current official benchmark concerns:

- `core_accuracy` -> `bench_harness.py`
- `api_speed` -> `benchmark_api_surface.py`
- `matrix_speed` -> `benchmark_matrix_suite.py`
- `matrix_compile` -> `benchmark_matrix_stack_diagnostics.py`
- `matrix_ad` -> `benchmark_matrix_free_krylov.py`
- `matrix_backend_compare` -> `benchmark_matrix_backend_candidates.py`
- `transform_speed` -> `benchmark_fft_nufft.py`
- `transform_backend_compare` -> `benchmark_nufft_backends.py`
- `transform_gpu` -> `benchmark_fft_nufft.py`

These official mappings are implemented in [taxonomy.py](/home/phili/projects/arbplusJAX/benchmarks/taxonomy.py).

## Shared Benchmark Output Schema

Benchmarks may continue to print human-readable summaries, but artifact output
should converge on the shared schema in [schema.py](/home/phili/projects/arbplusJAX/benchmarks/schema.py).

The shared record fields are:

- `benchmark_name`
- `concern`
- `category`
- `implementation`
- `operation`
- `device`
- `dtype`
- `cold_time_s`
- `warm_time_s`
- `recompile_time_s`
- `python_overhead_s`
- `memory_bytes`
- `accuracy_abs`
- `accuracy_rel`
- `residual`
- `ad_forward_time_s`
- `ad_backward_time_s`
- `ad_residual`

Additional measurements may be attached through structured measurement entries
instead of inventing one-off top-level JSON keys per script.

## CI Policy

Normal correctness CI should not run full benchmark sweeps.

Expected split:

- benchmark smoke in pytest:
  - CLI/help-path validation
  - taxonomy completeness
- real benchmark sweeps:
  - run through benchmark tools or dedicated harness profiles
  - write artifacts under `experiments/benchmarks/`

Benchmarks should fail in pytest only on explicit guardrails, not because a full
performance report exists.

## Matrix-Specific Rule

For matrix/operator benchmarks, every benchmark row should aim to separate:

- accuracy and residual behavior
- cold compile/startup cost
- warm runtime
- recompile sensitivity
- backend comparison

That split is especially important for:

- dense matrix surfaces
- sparse and cached sparse surfaces
- matrix-free/operator-plan paths
- PETSc/SLEPc comparison probes
- forward/backward matrix-free AD paths

## External Backend Rule

PETSc/SLEPc benchmark coverage is comparison-only.

It is valid to benchmark:

- PETSc sparse `Mat.mult`
- PETSc `KSP.solve`
- SLEPc `EPS.solve`

It is not valid to treat those timings as part of the governed runtime contract
for `src/arbplusjax`.
