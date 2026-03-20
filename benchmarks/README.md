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

- Use `tools/run_test_harness.py --profile bench-smoke` for lightweight benchmark smoke checks.
- Use `python -m pytest benchmarks/test_benchmark_smoke.py` for the direct benchmark CLI smoke path.
- Use `tools/run_benchmarks.py` or `tools/run_harness_profile.py` for real sweeps and reporting.
- Do not treat full benchmark sweeps as part of the normal correctness harness.

Primary entry points:
- `bench_harness.py`: sweep-based accuracy/speed comparison for JAX interval/point modes.
- `compare_*.py`: focused parity/accuracy checks.
- `benchmark_*.py`: focused throughput/timing runs.

Matrix-free benchmark entry points:
- `benchmark_matrix_free_krylov.py`: focused forward/backward timing for `jrb_mat` Lanczos and `jcb_mat` Arnoldi action/logdet paths.

Dense matrix benchmark entry points:
- `benchmark_dense_matrix_surface.py`: focused dense `arb_mat` / `acb_mat` timing for direct solve, LU-reuse solve, cached matvec, transpose, conjugate-transpose, and diagonal helpers.
  - example: `python benchmarks/benchmark_dense_matrix_surface.py --n 32 --runs 3`

Sparse matrix benchmark entry points:
- `benchmark_sparse_matrix_surface.py`: focused sparse `srb_mat` / `scb_mat` timing for COO, CSR, BCOO, and cached sparse matvec paths.

Every `bench_harness.py` run now writes `runtime_manifest.json` into the benchmark output directory so benchmark reports carry the same environment header schema as `tools/run_test_harness.py --outdir ...`.

Recommended invocations:
- Quick sweep: `python tools/run_benchmarks.py --profile quick`
- Full sweep: `python tools/run_benchmarks.py --profile full`
- Markdown report from latest run: `python tools/bench_report.py`

Optional Boost baseline:
- Pass `--boost-ref-cmd "<command>"` to `tools/run_benchmarks.py` or set `BOOST_REF_CMD`.
- The command must follow the stdin/stdout JSON contract documented in `boost_ref_adapter_template.py`.
