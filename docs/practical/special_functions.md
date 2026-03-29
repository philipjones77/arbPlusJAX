# Special Functions

Use the public special-function surfaces through three production patterns:

- point repeated-call service binders for stable-shape throughput
- interval/basic binders or family-owned mode wrappers when enclosure behavior matters
- family-owned diagnostics for scalar hardening decisions

The current canonical notebooks are:

- [example_gamma_family_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_gamma_family_surface.ipynb)
- [example_barnes_double_gamma_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_barnes_double_gamma_surface.ipynb)
- [example_hypgeom_family_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_hypgeom_family_surface.ipynb)

## Production Routes

For repeated point traffic, prefer:

```python
from arbplusjax import api

gamma_bound = api.bind_point_batch_jit(
    "incomplete_gamma_upper",
    dtype="float64",
    shape_bucket_multiple=128,
    method="quadrature",
    regularized=True,
    backend="auto",
)
```

For repeated basic traffic, prefer:

```python
gamma_basic = api.bind_interval_batch_jit_with_diagnostics(
    "incomplete_gamma_upper",
    mode="basic",
    dtype="float64",
    shape_bucket_multiple=128,
    prec_bits=53,
    method="quadrature",
    regularized=True,
    backend="auto",
)
```

For Barnes/double-gamma, treat the supported hardened route as:

- `ifj_barnesdoublegamma`
- `ifj_log_barnesdoublegamma`
- `ifj_barnesdoublegamma_diagnostics`

The legacy `bdg_*` route still exists, but the retained startup evidence should be read as lineage/reference coverage rather than the primary operational route.

## Diagnostics

Keep diagnostics off the hot path:

- binder diagnostics: `api.bind_point_batch_with_diagnostics(...)`, `api.bind_point_batch_jit_with_diagnostics(...)`
- family diagnostics: scalar `return_diagnostics=True` or explicit diagnostics functions like `ifj_barnesdoublegamma_diagnostics`

Use binder diagnostics for:

- backend choice
- effective padded/bucketed shape
- JIT/non-JIT route confirmation

Use family diagnostics for:

- method chosen
- fallback used
- truncation state
- recurrence / panel / shift counts

## CPU And GPU

Special functions are structurally fast-JAX on the supported repeated-call surfaces, but backend-realized performance is family-specific.

Current practical rule:

- CPU is the default for smaller service workloads and scalar diagnostics checks
- GPU should be considered for larger repeated stable-shape batches only after prewarm
- do not assume the Barnes legacy path is the right GPU target; use the supported IFJ/provider route when benchmarking operational startup or throughput
- backend-closeout claims for the current special-function tranche explicitly exclude Barnes IFJ batch throughput because that compile/runtime path is still under active hardening

The retained operational benchmark is:

- [benchmark_special_function_service_api.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_special_function_service_api.py)

Cross-family hardening and AD benchmarks are:

- [special_function_hardening_benchmark.py](/home/phili/projects/arbplusJAX/benchmarks/special_function_hardening_benchmark.py)
- [special_function_ad_benchmark.py](/home/phili/projects/arbplusJAX/benchmarks/special_function_ad_benchmark.py)

## AD

The production requirement for special functions is:

- argument-direction AD on the main evaluation variable
- parameter-direction AD on the family parameter when that parameter is continuous

The current audit/benchmark surfaces are:

- [test_special_function_ad_directions.py](/home/phili/projects/arbplusJAX/tests/test_special_function_ad_directions.py)
- [parameterized_ad_verification.md](/home/phili/projects/arbplusJAX/docs/reports/parameterized_ad_verification.md)
- [special_function_ad_benchmark.py](/home/phili/projects/arbplusJAX/benchmarks/special_function_ad_benchmark.py)

## What To Compare

For each family, compare:

- direct scalar value vs repeated-call point binder
- point vs basic/adaptive/rigorous when mode ownership exists
- public route vs family-owned exact-input/reference route
- CPU vs GPU only on the same stable-shape padded service call
- argument-direction AD vs parameter-direction AD

The current status rollup is:

- [special_function_status.md](/home/phili/projects/arbplusJAX/docs/reports/special_function_status.md)
