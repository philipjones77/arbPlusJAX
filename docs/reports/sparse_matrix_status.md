Last updated: 2026-03-29T00:00:00Z

# Sparse Matrix Status

## Current Status

Sparse point/basic operational apply surfaces are now explicitly hardened and
measured for CPU/GPU without dense fallback on the owned path.

Closed in this tranche:

- sparse real `point/basic` `matvec`
- sparse real `point/basic` `rmatvec`
- sparse real `point/basic` cached `prepare/apply`
- sparse complex `point/basic` `matvec`
- sparse complex `point/basic` `rmatvec`
- sparse complex `point/basic` cached `prepare/apply`
- point/basic compiled batch binders for sparse cached apply
- explicit no-dense-fallback regression tests on those surfaces

Open:

- sparse `basic` solve/factor/structured plan parity without dense lifting
- broader sparse diagnostics/status reporting beyond operational apply
- deeper sparse CPU/GPU crossover characterization for non-apply surfaces

## Validation

Primary tests:

- [test_sparse_point_api.py](/tests/test_sparse_point_api.py)
- [test_sparse_mode_surface.py](/tests/test_sparse_mode_surface.py)
- [test_sparse_basic_contracts.py](/tests/test_sparse_basic_contracts.py)
- [test_sparse_operational_contracts.py](/tests/test_sparse_operational_contracts.py)

Primary benchmark:

- [benchmark_sparse_operational_surface.py](/benchmarks/benchmark_sparse_operational_surface.py)
- [benchmark_sparse_operational_surface_cpu_refresh.json](/benchmarks/results/benchmark_sparse_operational_surface/benchmark_sparse_operational_surface_cpu_refresh.json)
- [benchmark_sparse_operational_surface_gpu_refresh.json](/benchmarks/results/benchmark_sparse_operational_surface/benchmark_sparse_operational_surface_gpu_refresh.json)

Canonical notebook:

- [example_sparse_matrix_surface.ipynb](/examples/example_sparse_matrix_surface.ipynb)

## Interpretation

The sparse tranche should now be read in two layers.

Operational sparse-native layer:

- point/basic cached sparse apply
- point/basic sparse `matvec`/`rmatvec`
- compiled point/basic batch binders

Broader sparse completion layer:

- factorization-backed solves
- structured sparse interval/box parity
- deeper spectral and selected-inverse work

The first layer is hardened in this pass. The second layer remains partially
open.

## Current CPU / GPU Readout

Retained sparse operational benchmark, `csr`, `n=32`, `float64`:

- `srb` point cached apply:
  - CPU `0.00217s`
  - GPU `0.01180s`
- `srb` point compiled cached batch apply:
  - CPU `0.00048s`
  - GPU `0.00087s`
- `scb` point compiled cached batch apply:
  - CPU `0.00055s`
  - GPU `0.00062s`

Interpretation:

- GPU is now validated on the sparse operational binder path.
- CPU still wins in the retained sparse real/complex operational slice.
- GPU gets closest on compiled sparse cached batch apply, especially complex.
- This is enough to claim sparse operational fast-JAX and operational-JAX
  coverage, but not enough to claim GPU-default superiority.
