Last updated: 2026-03-22T00:00:00Z

# Benchmark Grouping Standard

Status: active

## Purpose

This document defines how benchmark functionality is grouped in this repo.

Use it together with:

- [benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)

The goal is to keep benchmark organization consistent across:

- pytest markers
- benchmark CLI scripts
- benchmark artifacts
- benchmark reports

This document owns:

- allowed grouping axes
- allowed benchmark functionality groups
- allowed intent groups
- taxonomy/classification requirements

It does not define benchmark output schema details or benchmark measurement
policy. Those belong to
[benchmark_validation_policy_standard.md](/docs/standards/benchmark_validation_policy_standard.md).

## Grouping Axes

Every benchmark must be classified on three axes:

1. intent
   - what the benchmark is trying to measure
2. functionality category
   - what part of the repo the benchmark belongs to
3. device class
   - where the benchmark is expected to run

## Intent Groups

Use exactly one primary intent group per benchmark script:

- `accuracy`
  - error, residual, truth comparison
- `perf`
  - steady-state speed and general runtime behavior
- `compile`
  - cold compile and recompile behavior
- `ad`
  - forward/backward and autodiff cost
- `compare`
  - backend or external-software comparison

Pytest markers:

- `benchmark_accuracy`
- `benchmark_perf`
- `benchmark_compile`
- `benchmark_ad`
- `benchmark_compare`

## Functionality Groups

Every benchmark must also belong to one primary functionality group:

- `api`
- `scalar`
- `special`
- `combinatorics`
- `transform`
- `backend_matrix`
- `backend_transform`
- `matrix`
- `matrix_dense`
- `matrix_sparse`
- `matrix_block_sparse`
- `matrix_vblock_sparse`
- `matrix_free`

Pytest markers:

- `benchmark_api`
- `benchmark_scalar`
- `benchmark_special`
- `benchmark_combinatorics`
- `benchmark_transform`
- `benchmark_backend_matrix`
- `benchmark_backend_transform`
- `benchmark_matrix`
- `benchmark_matrix_dense`
- `benchmark_matrix_sparse`
- `benchmark_matrix_block_sparse`
- `benchmark_matrix_vblock_sparse`
- `benchmark_matrix_free`

## Device Groups

Every benchmark must declare its default device class:

- `cpu`
- `gpu_optional`

Pytest markers:

- `benchmark_cpu`
- `benchmark_gpu`

## Official Benchmark Rule

Some concerns have one official benchmark.

Official means:

- it is the canonical benchmark for regression discussion
- it is the default benchmark to cite in docs and reports
- it is the first benchmark to update when artifact schemas or harness policy change

Pytest marker:

- `benchmark_official`

Only designate an official benchmark when it is:

- representative
- stable
- diagnosable
- reproducible
- comparable

## Naming Rule

Benchmark scripts should be named so grouping is inferable from the filename
when possible:

- `benchmark_<family>.py`
- `benchmark_<family>_surface.py`
- `benchmark_<family>_service_api.py`
- `benchmark_<family>_diagnostics.py`
- `benchmark_<family>_backends.py`
- `compare_<family>.py`

Do not add benchmark scripts whose grouping cannot be reasonably inferred and
then confirmed in the taxonomy.

## Taxonomy Rule

The executable grouping source of truth is:

- [taxonomy.py](/benchmarks/taxonomy.py)

This standard defines the allowed grouping model.
The taxonomy defines the current repo mapping.

New benchmark entrypoints must not be added unless:

- they are classified in the taxonomy
- they fit one existing intent group
- they fit one existing functionality group
- they declare one default device class

## Reports Rule

Standards belong in `docs/standards`.
Current benchmark inventories and script-to-group mappings belong in `docs/reports`.
