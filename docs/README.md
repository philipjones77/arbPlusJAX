Last updated: 2026-03-01T00:00:00Z

# Documentation

This folder is organized by purpose (similar to `docs_continuous` style): governance at top level, implementation notes under `implementation/`, references under `references/`, and generated status reports under `reports/`.

- See `architecture.md` for the overall system layout and execution modes.
- See `benchmarks.md` and `benchmark_process.md` for benchmark backends, sweeps, and reporting workflow.
- See `engineering_policy.md` for the repo-wide implementation contract and engineering-status methodology.
- See `results/benchmarks/bessel_compile_probe_float32/bessel_compile_probe.md` for the current canonical Bessel padded-core compile probe.
- See `implementation/modules/hypgeom.md` for current hypergeometric scope and approximations.
- See `implementation/modules/jrb_mat.md` and `implementation/modules/jcb_mat.md` for the Jones-labeled matrix-function subsystem scaffold.
- See `implementation/external/cusf_compat.md` for the separate `cusf_*` compatibility implementation lineage and mapping.
- Use these notes as the source of truth for accuracy and parity expectations.
