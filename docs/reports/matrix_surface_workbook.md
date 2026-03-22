Last updated: 2026-03-20T00:00:00Z

# Matrix Surface Workbook

Use `python benchmarks/matrix_surface_workbook.py` to regenerate this report from the current dense, sparse, and matrix-free benchmark entry points.

The workbook is the summary layer over:

- `benchmarks/benchmark_dense_matrix_surface.py`
- `benchmarks/benchmark_sparse_matrix_surface.py`
- `benchmarks/benchmark_matrix_free_krylov.py`

The intent is to compare matrix families by execution strategy:

- dense direct kernels
- cached prepare/apply reuse paths
- sparse storage-family paths
- matrix-free operator-plan reuse paths

Regenerate before relying on any numbers in this file.
