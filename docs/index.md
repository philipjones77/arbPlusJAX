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
- `benchmarks.md`
- `audit.md`
- `todo.md`

Implementation docs:

- `implementation/modules/` for module families (`arb_*`, `acb_*`, `hypgeom`, etc.)
- `implementation/wrappers/` for wrapper and mode-dispatch docs
- `implementation/external/` for external-lineage implementations (`boost_*`, `Cusf_*`)

Reference docs:

- `references/bibliography/` for `.bib` files
- `references/inventory/function_list.md` for registry inventory

Generated reports:

- `reports/missing_impls/` for machine-generated missing-function and filtered lists
