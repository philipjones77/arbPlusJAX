Last updated: 2026-03-29T00:00:00Z

# Special-Function Status

Generated from `tools/special_function_status_report.py` using the checked-in hardening benchmark and startup-probe artifacts.

Scope:
- production-facing special-function families with active hardening work in this repo
- canonical API/binder surfaces, diagnostics surfaces, benchmarks, startup probes, and example notebooks

## Closeout Status

- special functions are now treated as closed for the governed production set excluding Barnes
- that closed set is: incomplete gamma, incomplete Bessel, and hypergeometric canonical surfaces
- Barnes / double-gamma remains the explicit exception backlog because its batched compiled throughput path is still compile-heavy

## Canonical Production Surfaces

| family | point surface | tighter/diagnostics surface | canonical notebook | primary tests | benchmark/startup evidence |
|---|---|---|---|---|---|
| incomplete gamma + incomplete Bessel I/K | `api.bind_point_batch()` / `api.bind_point_batch_jit()` on `incomplete_gamma_upper`, `incomplete_gamma_lower`, `incomplete_bessel_i`, `incomplete_bessel_k` | function-returned diagnostics via `return_diagnostics=True` plus `api.bind_interval_batch()` for stable-shape interval service usage | [examples/example_gamma_family_surface.ipynb](/examples/example_gamma_family_surface.ipynb) | [tests/test_incomplete_gamma.py](/tests/test_incomplete_gamma.py), [tests/test_incomplete_bessel_i.py](/tests/test_incomplete_bessel_i.py), [tests/test_special_function_service_contracts.py](/tests/test_special_function_service_contracts.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py) | [benchmarks/special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py) |
| Barnes / double-gamma | `double_gamma.ifj_barnesdoublegamma(...)` and explicit implementation-owned scalar/vector paths | `double_gamma.ifj_barnesdoublegamma_diagnostics(...)` and the Barnes G aliases on `acb_core` | [examples/example_barnes_double_gamma_surface.ipynb](/examples/example_barnes_double_gamma_surface.ipynb) | [tests/test_barnes_double_gamma_ifj_contracts.py](/tests/test_barnes_double_gamma_ifj_contracts.py), [tests/test_double_gamma_contracts.py](/tests/test_double_gamma_contracts.py), [tests/test_barnes_tier1.py](/tests/test_barnes_tier1.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py), [tests/test_special_function_ad_directions.py](/tests/test_special_function_ad_directions.py) | [benchmarks/benchmark_barnes_double_gamma.py](/benchmarks/benchmark_barnes_double_gamma.py), [benchmarks/double_gamma_point_startup_probe.py](/benchmarks/double_gamma_point_startup_probe.py), [benchmarks/special_function_ad_benchmark.py](/benchmarks/special_function_ad_benchmark.py) |
| hypergeom / regularized incomplete gamma | `api.bind_point_batch_jit("hypgeom.arb_hypgeom_1f1", ...)` and other family-owned point kernels | `api.bind_interval_batch(...)` for stable-shape interval routing plus `hypgeom_wrappers.*_mode*_padded` for direct family mode ownership | [examples/example_hypgeom_family_surface.ipynb](/examples/example_hypgeom_family_surface.ipynb) | [tests/test_hypgeom_wrappers_contracts.py](/tests/test_hypgeom_wrappers_contracts.py), [tests/test_hypgeom_engineering.py](/tests/test_hypgeom_engineering.py), [tests/test_hypgeom_startup_lazy_loading.py](/tests/test_hypgeom_startup_lazy_loading.py), [tests/test_special_function_hardening.py](/tests/test_special_function_hardening.py), [tests/test_special_function_ad_directions.py](/tests/test_special_function_ad_directions.py) | [docs/reports/hypgeom_status.md](/docs/reports/hypgeom_status.md), [benchmarks/hypgeom_point_startup_probe.py](/benchmarks/hypgeom_point_startup_probe.py), [benchmarks/special_function_hardening_benchmark.py](/benchmarks/special_function_hardening_benchmark.py), [benchmarks/special_function_ad_benchmark.py](/benchmarks/special_function_ad_benchmark.py) |

## Current Hardening Snapshot

- incomplete Bessel I quadrature: `0.635601`
- incomplete Bessel I high precision refine: `0.662488`
- incomplete Bessel I auto: `0.316868`
- Barnes IFJ vector: `26.874843`
- Barnes provider vector: `2.512507`
- Barnes BDG vector: `9.117547`
- Barnes IFJ recurrence residual: `0.0`
- Barnes provider vs IFJ max abs: `0.0`
- Barnes IFJ diagnostics m_used: `96`
- Barnes IFJ diagnostics n_shift: `6`
- Hypergeom 1f1 point batch: `0.139214`
- Hypergeom U point batch: `0.153482`
- Hypergeom U adaptive mode batch: `1.484077`
- Hypergeom U point vs scalar family max abs: `0.0`
- Hypergeom U mode vs scalar family max abs: `0.0`
- Hypergeom pfq point batch: `0.305530`
- Hypergeom pfq adaptive mode batch: `0.937578`
- Hypergeom pfq point vs scalar family max abs: `0.0`
- Hypergeom pfq mode vs scalar family max abs: `0.0`
- Hypergeom regularized lower batch: `2.226571`
- Hypergeom regularized upper batch: `1.848764`

## Operational Service Snapshot

- incomplete gamma upper point padded float64 warm: CPU `0.331140`, GPU `0.283432`
- incomplete gamma upper basic padded float64 warm: CPU `0.423250`, GPU `0.305107`
- incomplete Bessel K point padded float64 warm: CPU `0.846042`, GPU `0.315276`
- provider Barnes double-gamma padded float64 warm: CPU `n/a`, GPU `n/a`
- provider log Barnes double-gamma padded float64 warm: CPU `n/a`, GPU `n/a`

## Startup Probe Snapshot

- [hypgeom_point_startup_probe.json](/benchmarks/results/hypgeom_point_startup_probe/hypgeom_point_startup_probe.json): import=`0.830366`, compile+first=`0.150424`, steady=`0.000087`
- [double_gamma_point_startup_probe.json](/benchmarks/results/double_gamma_point_startup_probe/double_gamma_point_startup_probe.json) IFJ: import=`2.145601`, compile+first=`23.801830`, steady=`0.019230`
- provider Barnes startup: compile+first=`30.625186`, steady=`15.983087`
- legacy BDG startup: compile_error_type=`ConcretizationTypeError`

## Notes

- `special_function_hardening_benchmark.py` is the current cross-family benchmark/diagnostics rollup.
- `special_function_ad_benchmark.py` is the current cross-family argument-vs-parameter AD benchmark rollup.
- `benchmark_special_function_service_api.py` is the current repeated-call operational benchmark for point/basic service usage.
- `hypgeom_status.md` remains the detailed family-by-family hypergeom engineering inventory.
- The point-surface audit now compares public point wrappers against scalar family-owned exact-input evaluations; current remaining pfq drift is in the batched interval/mode path, not the point wrapper.
- The Barnes startup probe now records both the supported IFJ/provider startup path and the legacy BDG compile failure so the hardened route stays explicit.
- Current CPU/GPU closeout for the special-function tranche excludes Barnes IFJ batch throughput; gamma, incomplete Bessel, and hypergeom are the backend-certified focus set.
- On the retained padded service benchmark for that non-Barnes closeout set, GPU is ahead for the current incomplete-gamma and incomplete-Bessel comparison rows.
- Canonical notebook teaching is now split by ownership: gamma/incomplete-tail, Barnes/double-gamma, and hypergeom each have a dedicated example surface.
