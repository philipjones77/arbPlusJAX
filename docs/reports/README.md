Last updated: 2026-03-01T00:00:00Z

# Reports

Generated artifacts and audit outputs.

Refresh/verify command:
- `python tools/check_generated_reports.py`

- `missing_impls/`: missing implementation snapshots and filtered lists
- `core_function_status.md`: generated status table for the current `arb_core` / `acb_core` surface and mode coverage
- `core_point_status.md`: generated point-wrapper availability table for the current `arb_core` / `acb_core` surface
- `custom_core_status.md`: generated status table for custom core-complement functions and their tightening backlog
- `function_provenance_registry.md`: generated summary for canonical Arb-like, alternative, and new-function provenance classes
- `function_implementation_index.md`: generated index from base function names to all registered implementations
- `function_engineering_status.md`: generated engineering-status matrix for registered implementations, including current JAX, dtype, batching, AD, and hardening status
- `arb_like_functions.md`: generated registry rows for the canonical Arb-like public surface
- `alternative_functions.md`: generated registry rows for prefixed alternative implementations
- `new_functions.md`: generated registry rows for new mathematical families without an Arb-like base name
- `core_mode_benchmark_smoke.md`: warmed-JIT CPU smoke benchmark for selected real and complex core mode paths
- `../results/benchmarks/bessel_compile_probe_float32/bessel_compile_probe.md`: focused compile-count probe for canonical Bessel batch paths, including padded-core vs unpadded comparison
