Last updated: 2026-03-22T00:00:00Z

# Core Scalar Service Calling Standard

Status: active

## Purpose

This standard defines how `1. Core Numeric Scalars` should be called in real
service or worker processes.

This is the canonical professional-JAX calling model for the core scalar
tranche.

Use it together with:

- [jax_api_runtime_standard.md](/home/phili/projects/arbplusJAX/docs/standards/jax_api_runtime_standard.md)
- [environment_portability_standard.md](/home/phili/projects/arbplusJAX/docs/standards/environment_portability_standard.md)
- [test_coverage_matrix.md](/home/phili/projects/arbplusJAX/docs/status/test_coverage_matrix.md)

## Scope

This standard applies to:

- `arb_core`
- `acb_core`
- `arf`
- `acf`
- `fmpr`
- `fmpzi`
- `arb_fpwrap`

when used through the public API and runtime configuration surfaces.

## Canonical Production Pattern

The default production pattern is:

1. long-lived worker
2. public API entrypoint
3. fixed dtype policy
4. fixed padded or bucketed shapes
5. warmup phase
6. persistent compilation cache
7. latency/throughput/recompile measurement in `benchmarks/`
8. correctness/robustness validation in `tests/`

This means the canonical service model is not:

- notebook-first
- raw-kernel-first
- arbitrary-shape-first

## Public API Rule

Continuous-call service code should use the real public API surface:

- `api.eval_point(...)`
- `api.eval_point_batch(...)`
- `api.eval_interval(...)`
- `api.eval_interval_batch(...)`
- `api.bind_point(...)`
- `api.bind_point_batch(...)`
- `api.bind_interval(...)`
- `api.bind_interval_batch(...)`
- `api.evaluate(...)` where routing is part of the contract

For repeated worker calls, prefer the bound-call surfaces so operation choice,
dtype policy, and optional batch settings are fixed once per worker.

Recommended example:

```python
from arbplusjax import api

service_fn = api.bind_point_batch(
    "fmpr_mul",
    dtype="float32",
    pad_to=4096,
)
```

Chunked worker example:

```python
from arbplusjax import api

service_fn = api.bind_point_batch(
    "arb_fpwrap_double_exp",
    dtype="float64",
    pad_to=4096,
    chunk_size=1024,
)
```

## Dtype Policy Rule

Every worker should choose one explicit dtype policy up front:

- `float32`
- `float64`

Do not let mixed dtypes drift through a hot worker process.

The worker should either:

- bind `dtype="float32"`
- bind `dtype="float64"`

and then keep that policy stable for the lifetime of the worker.

## Shape Policy Rule

Do not let arbitrary batch shapes hit a hot worker if you care about latency
stability.

Instead:

- define a small number of batch-size buckets
- pad requests into those buckets
- keep `pad_to` stable for the workload class

This is the default strategy for reducing recompilation churn.

## Warmup Rule

A worker should warm the hot call surfaces once before serving live traffic.

Warmup should:

- touch each expected shape bucket
- touch each expected dtype policy
- populate the in-process JIT cache
- populate the persistent compile cache when enabled

## Persistent Cache Rule

Service and benchmark-entry worker processes should enable persistent
compilation cache when repeated restarts are expected.

Recommended env policy:

- `JAX_ENABLE_COMPILATION_CACHE=1`
- `JAX_COMPILATION_CACHE_DIR=<stable writable path>`

Do not mix unrelated backends or incompatible toolchains into one cache path.

## Optional Parameter Rule

Optional batch/runtime parameters that should be bound intentionally rather than
decided ad hoc at every call include:

- `dtype`
- `pad_to`
- `chunk_size`
- `mode`
- `prec_bits`
- `dps`

If a harder function later adds method/strategy options, those should also be
bound once per worker whenever practical.

## Benchmark Placement Rule

Service-like speed and bottleneck measurement belongs in `benchmarks/`.

Examples:

- padded vs unpadded batch cost
- p50/p95/p99 repeated-call latency
- compile versus warm versus recompile behavior
- restart plus persistent-cache behavior
- memory drift and throughput soak runs

These do not belong primarily in notebooks.

## Test Placement Rule

Correctness and robustness belong in `tests/`.

Examples:

- repeated-call correctness
- malformed input handling
- shape-bucket and padding invariants
- dtype-policy invariants
- restart-safe contract behavior

These should be runnable under `pytest`.

## Example Placement Rule

Notebooks are demonstration and reporting layers.

They should:

- show the canonical service-style API call shape
- summarize retained benchmark/test artifacts
- visualize latency and compile behavior

They should not be treated as the primary production execution surface.

## Core Scalar Default

For Core Numeric Scalars specifically, the default real-life usage model should
be:

- a long-lived process
- a bound public API function
- one explicit dtype policy
- one explicit shape bucket or padding policy per workload class
- warmup before steady-state calls

That is the default professional setup for this tranche.
