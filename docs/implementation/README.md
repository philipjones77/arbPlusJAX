Last updated: 2026-03-17T00:00:00Z

# Implementation Docs

- implementation-facing notes:
  - `jax_setup.md`
  - `stable_kernel_subset.md`
- existing implementation-tree documents:
  - `build.md`
  - `linux_gpu_colab.md`
  - `run_platform.md`
  - `benchmarks.md`
  - `benchmark_process.md`
  - `testing_harness.md`
  - `matrix_logdet_landscape.md`
  - `precision_guardrails_gpu.md`
  - `soft_ops_optional.md`
- `modules/`: module-level implementation notes (`arb_*`, `acb_*`, `hypgeom`, etc.)
  - includes `modules/dft.md`
  - includes `modules/nufft.md`
- `wrappers/`: wrapper/mode-dispatch notes and interval API behavior
- `external/`: separate implementation lineages integrated into this workspace
  - `ducc_review.md`
  - `parsinv_review.md`

For practical runbooks, benchmarking guidance, and numerically informed operating notes, start from [docs/practical/README.md](/home/phili/projects/arbplusJAX/docs/practical/README.md).

The practical layer is additive. Most current documentation material remains in `docs/implementation/`, and `docs/practical/` exists to provide separate run/use guidance rather than to rename this tree.
