Last updated: 2026-03-01T00:00:00Z

# arbPlusJAX docs

This folder uses a stable layout:

- top-level governance docs (`README.md`, `index.md`, `todo.md`, `audit.md`, `architecture.md`)
- implementation docs in `implementation/`
- references in `references/`
- generated status artifacts in `reports/`

Top-level docs:

- `README.md`
- `architecture.md`
- `build.md`
- `jax_setup.md`
- `precision.md`
- `function_naming.md`
- `engineering_policy.md`
- `benchmarks.md`
- `benchmark_process.md`
- `audit.md`
- `todo.md`

Implementation docs:

- `implementation/modules/` for module families (`arb_*`, `acb_*`, `hypgeom`, etc.)
- `implementation/modules/jrb_mat.md` and `implementation/modules/jcb_mat.md` for the Jones-labeled matrix-function subsystem scaffold
- `implementation/wrappers/` for wrapper and mode-dispatch docs
- `implementation/external/` for external-lineage implementations (`boost_*`, `cusf_*`)

Reference docs:

- `references/bibliography/` for `.bib` files
- `references/inventory/function_list.md` for registry inventory

Generated reports:

- `reports/missing_impls/` for machine-generated missing-function and filtered lists
- `reports/core_function_status.md` for the generated core-surface implementation/mode table
- `reports/core_point_status.md` for the generated core-surface point-wrapper table
- `reports/custom_core_status.md` for the generated custom core-complement status and tightening table
- `reports/function_provenance_registry.md` for the generated naming/provenance summary
- `reports/function_implementation_index.md` for base-name lookup across canonical and alternative implementations
- `reports/function_engineering_status.md` for the generated current engineering-status matrix
- `results/benchmarks/bessel_compile_probe_float32/bessel_compile_probe.md` for the current canonical Bessel compile-count probe and padded-core comparison
- `reports/arb_like_functions.md` for canonical Arb-like public names
- `reports/alternative_functions.md` for prefixed alternative implementations
- `reports/new_functions.md` for new mathematical families without an Arb-like base name
- `reports/core_mode_benchmark_smoke.md` for a direct core-mode timing/width smoke benchmark
