# Benchmarks

`benchmarks/` is the implementation and CLI home for benchmark and cross-backend
comparison scripts.

It also remains the repo-root benchmark package used by the pytest-facing smoke
and harness entrypoints. `experiments/benchmarks/` does not replace it.

Benchmark experiments and their generated artifacts are canonicalized under
`experiments/benchmarks/`:

- run trees: `experiments/benchmarks/results/`
- generated reports and diagnostics: `experiments/benchmarks/outputs/`

This directory is intentionally separate from `tests/`.

Benchmark governance and pytest marker policy:

- [benchmark_validation_policy.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_validation_policy.md)
- [benchmark_grouping_standard.md](/home/phili/projects/arbplusJAX/docs/standards/benchmark_grouping_standard.md)
- [environment_portability_standard.md](/home/phili/projects/arbplusJAX/docs/standards/environment_portability_standard.md)
- taxonomy and marker derivation: [taxonomy.py](/home/phili/projects/arbplusJAX/benchmarks/taxonomy.py)
  - this also defines the official benchmark mapping for major benchmark concerns
- shared artifact schema: [schema.py](/home/phili/projects/arbplusJAX/benchmarks/schema.py)
- current grouped inventory: [benchmark_group_inventory.md](/home/phili/projects/arbplusJAX/docs/reports/benchmark_group_inventory.md)
- current portability inventory: [environment_portability_inventory.md](/home/phili/projects/arbplusJAX/docs/reports/environment_portability_inventory.md)

- Use `tools/run_test_harness.py --profile bench-smoke` for lightweight benchmark smoke checks.
- Use `python -m pytest benchmarks/test_benchmark_smoke.py` for the direct benchmark CLI smoke path.
- Use pytest markers to target benchmark intent/category slices, for example:
  - `pytest -m benchmark_perf`
  - `pytest -m benchmark_compile`
  - `pytest -m benchmark_ad`
  - `pytest -m "benchmark_compare and benchmark_matrix"`
  - `pytest -m benchmark_official`
  - `pytest -m "benchmark_transform and benchmark_gpu"`
- Use `tools/run_benchmarks.py` or `tools/run_harness_profile.py` for real sweeps and reporting.
- Do not treat full benchmark sweeps as part of the normal correctness harness.

Primary entry points:
- `bench_harness.py`: sweep-based accuracy/speed comparison for JAX interval/point modes.
- `compare_*.py`: focused parity/accuracy checks.
- `benchmark_*.py`: focused throughput/timing runs.

Matrix-free benchmark entry points:
- `benchmark_matrix_free_krylov.py`: focused forward/backward timing for `jrb_mat` Lanczos and `jcb_mat` Arnoldi action/logdet paths.
  - reports cold first-call timings separately from warmed plan-backed timings
  - enables JAX compilation cache and `x64` by default for repeatable benchmark sessions
  - supports `--sections real,sparse_real,complex,sparse_complex` so expensive slices can be isolated and rerun without paying for the full matrix-free suite
- `benchmark_matrix_stack_diagnostics.py`: focused compile/steady-state/recompile profiling for representative dense, sparse, and operator-plan kernels.
  - writes JSON diagnostics under `experiments/benchmarks/outputs/diagnostics/`
  - supports `--cases name1,name2,...` so compile-heavy kernels can be bisected directly
- `benchmark_api_surface.py`: focused routed-API overhead checks against direct API calls for scalar, incomplete-gamma, and matrix dispatch paths.
- `benchmark_matrix_backend_candidates.py`: optional cross-backend candidate matrix benchmarks covering dense JAX/JAX-Scipy, SciPy dense/sparse, `jax.experimental.sparse`, matrix-free `jrb_mat`, and environment-probed PETSc/SLEPc availability.

Dense matrix benchmark entry points:
- `benchmark_dense_matrix_surface.py`: focused dense `arb_mat` / `acb_mat` timing for direct solve, LU-reuse solve, cached matvec, transpose, conjugate-transpose, and diagonal helpers.
  - example: `python benchmarks/benchmark_dense_matrix_surface.py --n 32 --runs 3`

Sparse matrix benchmark entry points:
- `benchmark_sparse_matrix_surface.py`: focused sparse `srb_mat` / `scb_mat` timing for COO, CSR, BCOO, and cached sparse matvec paths.

Transform benchmark entry points:
- `benchmark_fft_nufft.py`: focused internal FFT/NUFFT throughput for direct, cached, batched, and ND transform paths.
- `benchmark_nufft_backends.py`: optional cross-backend NUFFT comparison covering repo-native JAX NUFFT plus environment-probed `nufftax` and `jax_finufft`.

Every `bench_harness.py` run now writes `runtime_manifest.json` into the benchmark output directory so benchmark reports carry the same environment header schema as `tools/run_test_harness.py --outdir ...`.

Recommended invocations:
- Quick sweep: `python tools/run_benchmarks.py --profile quick`
- Full sweep: `python tools/run_benchmarks.py --profile full`
- Markdown report from latest run: `python tools/bench_report.py`
- Matrix workbook report: `python tools/matrix_surface_workbook.py`

Optional Boost baseline:
- Pass `--boost-ref-cmd "<command>"` to `tools/run_benchmarks.py` or set `BOOST_REF_CMD`.
- The command must follow the stdin/stdout JSON contract documented in `boost_ref_adapter_template.py`.
