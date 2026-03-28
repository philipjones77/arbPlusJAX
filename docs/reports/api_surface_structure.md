Last updated: 2026-03-28T00:00:00Z

# API Surface Structure

This generated report consolidates the public `arbplusjax.api` surface into one place.

Use it as the practical companion to:
- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)
- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)
- [api_usability_standard.md](/docs/standards/api_usability_standard.md)

Interpretation:
- `direct` means an ordinary public evaluation surface.
- `bound service` means a repeated-call binder or bound callable surface.
- `compiled bound` means the repeated-call binder owns the compiled entrypoint explicitly.
- `diagnostics-bearing binder` means the bound surface returns both value and structured diagnostics.
- `policy helper` means the surface recommends or prepares the repeated-call policy rather than directly evaluating the function.

Total public `api` exports: `69`

Section counts:
- `Unified Routing`: 1
- `Direct Evaluation`: 6
- `Specialized Public Function Surfaces`: 30
- `Bound Service Surfaces`: 6
- `Compiled And AD Surfaces`: 7
- `Policy Helpers`: 4
- `Metadata And Registry Surfaces`: 6
- `Structured Payload Types`: 9

## Common Option Groups

| surface | key options |
|---|---|
| `bind_point_batch` | `dtype, pad_to, shape_bucket_multiple, chunk_size, backend, min_gpu_batch_size, prewarm` |
| `bind_point_batch_with_diagnostics` | `dtype, pad_to, shape_bucket_multiple, chunk_size, backend, min_gpu_batch_size, prewarm` |
| `bind_point_batch_jit` | `dtype, pad_to, shape_bucket_multiple, backend, min_gpu_batch_size, prewarm` |
| `bind_point_batch_jit_with_diagnostics` | `dtype, pad_to, shape_bucket_multiple, backend, min_gpu_batch_size, prewarm` |
| `bind_interval_batch` | `mode, prec_bits, dps, dtype, pad_to, shape_bucket_multiple, chunk_size, backend, min_gpu_batch_size, prewarm` |
| `bind_interval_batch_with_diagnostics` | `mode, prec_bits, dps, dtype, pad_to, shape_bucket_multiple, chunk_size, backend, min_gpu_batch_size, prewarm` |
| `bind_interval_batch_jit` | `mode, prec_bits, dps, dtype, pad_to, shape_bucket_multiple, backend, min_gpu_batch_size, prewarm` |
| `bind_interval_batch_jit_with_diagnostics` | `mode, prec_bits, dps, dtype, pad_to, shape_bucket_multiple, backend, min_gpu_batch_size, prewarm` |
| `choose_point_batch_policy` | `batch_size, dtype, backend, pad_to, shape_bucket_multiple, chunk_size, min_gpu_batch_size, prewarm` |
| `choose_interval_batch_policy` | `batch_size, dtype, mode, prec_bits, dps, backend, pad_to, shape_bucket_multiple, chunk_size, min_gpu_batch_size, prewarm` |
| `prewarm_core_point_kernels` | `names, dtype, backend, batch_size, shape_bucket_multiple, min_gpu_batch_size` |
| `prewarm_interval_mode_kernels` | `names, dtype, prec_bits, dps, backend, batch_size, shape_bucket_multiple, min_gpu_batch_size` |

## Unified Routing

| public_name | kind | signature |
|---|---|---|
| `evaluate` | `auto-routing` | `(name: 'str', *args: 'jax.Array', mode: 'str' = 'point', dtype: 'str \| jnp.dtype \| None' = None, value_kind: 'str \| None' = None, implementation: 'str \| None' = None, implementation_version: 'str \| None' = None, method: 'str \| None' = None, strategy: 'str \| None' = None, method_params: 'Mapping[str, object] \| None' = None, prec_bits: 'int \| None' = None, dps: 'int \| None' = None, **kwargs) -> 'jax.Array'` |

## Direct Evaluation

| public_name | kind | signature |
|---|---|---|
| `eval_point` | `direct` | `(name: 'str', *args: 'jax.Array', jit: 'bool' = False, dtype: 'str \| jnp.dtype \| None' = None, **kwargs) -> 'jax.Array'` |
| `eval_point_batch` | `direct` | `(name: 'str', *args: 'jax.Array', dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, **kwargs) -> 'jax.Array'` |
| `eval_point_batch_chunked` | `direct` | `(name: 'str', *args: 'jax.Array', chunk_size: 'int' = 1024, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, **kwargs) -> 'jax.Array'` |
| `eval_interval` | `direct` | `(name: 'str', *args: 'jax.Array', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, jit: 'bool' = False, dtype: 'str \| jnp.dtype \| None' = None, **kwargs) -> 'jax.Array'` |
| `eval_interval_batch` | `direct` | `(name: 'str', *args: 'jax.Array', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, **kwargs) -> 'jax.Array'` |
| `eval_interval_batch_chunked` | `direct` | `(name: 'str', *args: 'jax.Array', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, chunk_size: 'int' = 1024, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, **kwargs) -> 'jax.Array'` |

## Specialized Public Function Surfaces

| public_name | kind | signature |
|---|---|---|
| `tail_integral` | `specialized direct` | `(integrand_or_problem, lower_limit: 'jax.Array \| float \| None' = None, *, panel_width: 'float' = 0.25, max_panels: 'int' = 128, samples_per_panel: 'int' = 32, return_diagnostics: 'bool' = False) -> 'jax.Array \| tuple[jax.Array, TailEvaluationDiagnostics]'` |
| `tail_integral_batch` | `specialized direct` | `(integrand_or_problem, lower_limit, *, panel_width: 'float' = 0.25, max_panels: 'int' = 128, samples_per_panel: 'int' = 32)` |
| `tail_integral_accelerated` | `specialized direct` | `(integrand_or_problem, lower_limit: 'jax.Array \| float \| None' = None, *, method: 'str' = 'auto', panel_width: 'float' = 0.25, max_panels: 'int' = 128, samples_per_panel: 'int' = 32, recurrence: 'TailRatioRecurrence \| None' = None, derivative_metadata: 'TailDerivativeMetadata \| None' = None, regime_metadata: 'TailRegimeMetadata \| None' = None, return_diagnostics: 'bool' = False) -> 'jax.Array \| tuple[jax.Array, TailEvaluationDiagnostics]'` |
| `tail_integral_accelerated_batch` | `specialized direct` | `(integrand_or_problem, lower_limit, *, method: 'str' = 'auto', panel_width: 'float' = 0.25, max_panels: 'int' = 128, samples_per_panel: 'int' = 32, recurrence: 'TailRatioRecurrence \| None' = None, derivative_metadata: 'TailDerivativeMetadata \| None' = None, regime_metadata: 'TailRegimeMetadata \| None' = None)` |
| `incomplete_gamma_lower` | `specialized direct` | `(s, z, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24, return_diagnostics: 'bool' = False)` |
| `incomplete_gamma_lower_argument_derivative` | `specialized direct` | `(s, z, *, regularized: 'bool' = False)` |
| `incomplete_gamma_lower_batch` | `specialized direct` | `(s, z, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_gamma_lower_derivative` | `specialized direct` | `(s, z, *, respect_to: 'str' = 'z', regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_gamma_lower_parameter_derivative` | `specialized direct` | `(s, z, *, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_gamma_upper` | `specialized direct` | `(s, z, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24, return_diagnostics: 'bool' = False)` |
| `incomplete_gamma_upper_argument_derivative` | `specialized direct` | `(s, z, *, regularized: 'bool' = False)` |
| `incomplete_gamma_upper_batch` | `specialized direct` | `(s, z, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_gamma_upper_derivative` | `specialized direct` | `(s, z, *, respect_to: 'str' = 'z', regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_gamma_upper_parameter_derivative` | `specialized direct` | `(s, z, *, regularized: 'bool' = False, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `laplace_bessel_k_tail` | `specialized direct` | `(nu, z, lam, lower_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24, return_diagnostics: 'bool' = False)` |
| `laplace_bessel_k_tail_batch` | `specialized direct` | `(nu, z, lam, lower_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `laplace_bessel_k_tail_derivative` | `specialized direct` | `(nu, z, lam, lower_limit, *, respect_to: 'str' = 'lambda', method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `laplace_bessel_k_tail_lambda_derivative` | `specialized direct` | `(nu, z, lam, lower_limit, *, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `laplace_bessel_k_tail_lower_limit_derivative` | `specialized direct` | `(nu, z, lam, lower_limit)` |
| `incomplete_bessel_i` | `specialized direct` | `(nu, z, upper_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_count: 'int' = 128, samples_per_panel: 'int' = 16, return_diagnostics: 'bool' = False)` |
| `incomplete_bessel_i_batch` | `specialized direct` | `(nu, z, upper_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_count: 'int' = 128, samples_per_panel: 'int' = 16)` |
| `incomplete_bessel_i_derivative` | `specialized direct` | `(nu, z, upper_limit, *, respect_to: 'str' = 'z', method: 'str' = 'quadrature', panel_count: 'int' = 128, samples_per_panel: 'int' = 16)` |
| `incomplete_bessel_i_argument_derivative` | `specialized direct` | `(nu, z, upper_limit, *, method: 'str' = 'quadrature', panel_count: 'int' = 128, samples_per_panel: 'int' = 16)` |
| `incomplete_bessel_i_upper_limit_derivative` | `specialized direct` | `(nu, z, upper_limit, *, method: 'str' = 'quadrature', panel_count: 'int' = 128, samples_per_panel: 'int' = 16)` |
| `incomplete_bessel_k` | `specialized direct` | `(nu, z, lower_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24, return_diagnostics: 'bool' = False)` |
| `incomplete_bessel_k_batch` | `specialized direct` | `(nu, z, lower_limit, *, mode: 'str' = 'point', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_bessel_k_derivative` | `specialized direct` | `(nu, z, lower_limit, *, respect_to: 'str' = 'z', method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_bessel_k_argument_derivative` | `specialized direct` | `(nu, z, lower_limit, *, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `incomplete_bessel_k_lower_limit_derivative` | `specialized direct` | `(nu, z, lower_limit, *, method: 'str' = 'quadrature', panel_width: 'float' = 0.125, max_panels: 'int' = 160, samples_per_panel: 'int' = 24)` |
| `TailRatioRecurrence` | `specialized direct` | `(a0: 'float' = 0.0, a1: 'float' = 1.0, b0: 'float' = 1.0, b1: 'float' = 1.0, alpha: 'Callable[[int], float] \| None' = None, beta: 'Callable[[int], float] \| None' = None, gamma: 'Callable[[int], float] \| None' = None, delta: 'Callable[[int], float] \| None' = None, a_init: 'tuple[float, ...]' = (), b_init: 'tuple[float, ...]' = (), a_coeffs: 'Callable[[int], tuple[float, ...]] \| None' = None, b_coeffs: 'Callable[[int], tuple[float, ...]] \| None' = None, order: 'int' = 2, note: 'str' = '') -> None` |

## Bound Service Surfaces

| public_name | kind | signature |
|---|---|---|
| `bind_point` | `bound service` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, **bound_kwargs) -> 'Callable'` |
| `bind_point_batch` | `bound service` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, chunk_size: 'int \| None' = None, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_point_batch_with_diagnostics` | `diagnostics-bearing binder` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, chunk_size: 'int \| None' = None, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_interval` | `bound service` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, **bound_kwargs) -> 'Callable'` |
| `bind_interval_batch` | `bound service` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, chunk_size: 'int \| None' = None, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_interval_batch_with_diagnostics` | `diagnostics-bearing binder` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, chunk_size: 'int \| None' = None, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |

## Compiled And AD Surfaces

| public_name | kind | signature |
|---|---|---|
| `bind_point_batch_jit` | `compiled bound` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_point_batch_jit_with_diagnostics` | `diagnostics-bearing binder` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_point_jit` | `compiled bound` | `(name: 'str', dtype: 'str \| jnp.dtype \| None' = None, **bound_kwargs) -> 'Callable'` |
| `bind_interval_jit` | `compiled bound` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, **bound_kwargs) -> 'Callable'` |
| `bind_point_ad` | `compiled AD binder` | `(name: 'str', kind: 'str' = 'grad', argnums: 'int \| tuple[int, ...]' = 0, dtype: 'str \| jnp.dtype \| None' = None) -> 'Callable'` |
| `bind_interval_batch_jit` | `compiled bound` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |
| `bind_interval_batch_jit_with_diagnostics` | `diagnostics-bearing binder` | `(name: 'str', mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, pad_value: 'float \| complex' = 0.0, backend: 'str' = 'auto', min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False, **bound_kwargs) -> 'Callable'` |

## Policy Helpers

| public_name | kind | signature |
|---|---|---|
| `choose_point_batch_policy` | `policy helper` | `(*, batch_size: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, backend: 'str' = 'auto', pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, chunk_size: 'int \| None' = None, min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False) -> 'PointBatchPolicy'` |
| `choose_interval_batch_policy` | `policy helper` | `(*, batch_size: 'int \| None' = None, dtype: 'str \| jnp.dtype \| None' = None, mode: 'str' = 'basic', prec_bits: 'int \| None' = None, dps: 'int \| None' = None, backend: 'str' = 'auto', pad_to: 'int \| None' = None, shape_bucket_multiple: 'int \| None' = None, chunk_size: 'int \| None' = None, min_gpu_batch_size: 'int' = 2048, prewarm: 'bool' = False) -> 'IntervalBatchPolicy'` |
| `prewarm_core_point_kernels` | `policy helper` | `(names: 'tuple[str, ...] \| None' = None, *, dtype: 'str \| jnp.dtype \| None' = 'float64', backend: 'str' = 'auto', batch_size: 'int' = 1024, shape_bucket_multiple: 'int \| None' = 128, min_gpu_batch_size: 'int' = 2048) -> 'dict[str, PointBatchCallDiagnostics]'` |
| `prewarm_interval_mode_kernels` | `policy helper` | `(names: 'tuple[tuple[str, str], ...] \| None' = None, *, dtype: 'str \| jnp.dtype \| None' = 'float64', prec_bits: 'int \| None' = 53, dps: 'int \| None' = None, backend: 'str' = 'auto', batch_size: 'int' = 256, shape_bucket_multiple: 'int \| None' = 64, min_gpu_batch_size: 'int' = 512) -> 'dict[str, IntervalBatchCallDiagnostics]'` |

## Metadata And Registry Surfaces

| public_name | kind | signature |
|---|---|---|
| `list_public_functions` | `metadata/registry` | `() -> 'list[str]'` |
| `list_point_functions` | `metadata/registry` | `() -> 'list[str]'` |
| `list_interval_functions` | `metadata/registry` | `() -> 'list[str]'` |
| `get_public_function_metadata` | `metadata/registry` | `(name: 'str') -> 'PublicFunctionMetadata'` |
| `list_public_function_metadata` | `metadata/registry` | `(*, family: 'str \| None' = None, stability: 'str \| None' = None, module: 'str \| None' = None, name_prefix: 'str \| None' = None, derivative_status: 'str \| None' = None) -> 'list[PublicFunctionMetadata]'` |
| `render_public_function_metadata_json` | `metadata/registry` | `(*, family: 'str \| None' = None, stability: 'str \| None' = None, module: 'str \| None' = None, name_prefix: 'str \| None' = None, derivative_status: 'str \| None' = None) -> 'str'` |

## Structured Payload Types

| public_name | kind | signature |
|---|---|---|
| `PublicFunctionMetadata` | `structured payload` | `(name: 'str', qualified_name: 'str', module: 'str', implementation_name: 'str', family: 'str', stability: 'str', point_support: 'bool', interval_support: 'bool', interval_modes: 'tuple[str, ...]', value_kinds: 'tuple[str, ...]', implementation_options: 'tuple[str, ...]', implementation_versions: 'tuple[str, ...]', default_implementation: 'str', method_tags: 'tuple[str, ...]', default_method: 'str \| None', method_parameter_names: 'tuple[str, ...]', execution_strategies: 'tuple[str, ...]', regime_tags: 'tuple[str, ...]', derivative_status: 'str', notes: 'str') -> None` |
| `PointBatchPolicy` | `structured payload` | `(requested_backend: 'str', chosen_backend: 'str', batch_size: 'int \| None', dtype: 'str \| None', pad_to: 'int \| None', shape_bucket_multiple: 'int \| None', effective_pad_to: 'int \| None', chunk_size: 'int \| None', min_gpu_batch_size: 'int', prewarm: 'bool') -> None` |
| `PointBatchCallDiagnostics` | `structured payload` | `(name: 'str', requested_backend: 'str', chosen_backend: 'str', batch_size: 'int \| None', dtype: 'str \| None', pad_to: 'int \| None', shape_bucket_multiple: 'int \| None', effective_pad_to: 'int \| None', chunk_size: 'int \| None', jit_enabled: 'bool', compiled_this_call: 'bool', prewarmed: 'bool') -> None` |
| `IntervalBatchPolicy` | `structured payload` | `(requested_backend: 'str', chosen_backend: 'str', batch_size: 'int \| None', dtype: 'str \| None', mode: 'str', prec_bits: 'int \| None', dps: 'int \| None', pad_to: 'int \| None', shape_bucket_multiple: 'int \| None', effective_pad_to: 'int \| None', chunk_size: 'int \| None', min_gpu_batch_size: 'int', prewarm: 'bool') -> None` |
| `IntervalBatchCallDiagnostics` | `structured payload` | `(name: 'str', requested_backend: 'str', chosen_backend: 'str', batch_size: 'int \| None', dtype: 'str \| None', mode: 'str', prec_bits: 'int \| None', dps: 'int \| None', pad_to: 'int \| None', shape_bucket_multiple: 'int \| None', effective_pad_to: 'int \| None', chunk_size: 'int \| None', jit_enabled: 'bool', compiled_this_call: 'bool', prewarmed: 'bool') -> None` |
| `TailDerivativeMetadata` | `structured payload` | `(argument_derivative: 'bool' = False, lower_limit_derivative: 'bool' = False, parameter_derivative: 'bool' = False, note: 'str' = '') -> None` |
| `TailEvaluationDiagnostics` | `structured payload` | `(method: 'str', chunk_count: 'int', panel_count: 'int', recurrence_steps: 'int', estimated_tail_remainder: 'float', instability_flags: 'tuple[str, ...]' = (), fallback_used: 'bool' = False, precision_warning: 'bool' = False, note: 'str' = '') -> None` |
| `TailIntegralProblem` | `structured payload` | `(integrand: 'Callable[[jax.Array], jax.Array]', lower_limit: 'float \| jax.Array', panel_width: 'float' = 0.25, max_panels: 'int' = 128, samples_per_panel: 'int' = 32, quadrature_rule: 'str' = 'simpson', recurrence: 'TailRatioRecurrence \| None' = None, derivative_metadata: 'TailDerivativeMetadata' = TailDerivativeMetadata(argument_derivative=False, lower_limit_derivative=False, parameter_derivative=False, note=''), regime_metadata: 'TailRegimeMetadata' = TailRegimeMetadata(decay_rate=None, oscillation_level=None, near_singularity=False, cancellation_risk=False, note=''), name: 'str' = 'tail_integral') -> None` |
| `TailRegimeMetadata` | `structured payload` | `(decay_rate: 'float \| None' = None, oscillation_level: 'float \| None' = None, near_singularity: 'bool' = False, cancellation_risk: 'bool' = False, note: 'str' = '') -> None` |
