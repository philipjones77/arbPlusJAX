Last updated: 2026-03-25T00:00:00Z

# Dense Matrix Surface Benchmark

This report tracks the canonical dense benchmark entrypoint:

- [benchmark_dense_matrix_surface.py](/benchmarks/benchmark_dense_matrix_surface.py)

Current scope:

- dense real interval matrices in `arb_mat`
- dense complex box matrices in `acb_mat`
- direct solve vs LU-plan reuse
- dense cached `matvec` and padded cached `matvec`
- transpose / conjugate-transpose / diagonal extraction
- structured SPD / HPD solve-plan reuse
- block assembly and block-diagonal construction

Runtime contract:

- CPU is the active validation slice here
- CLI is explicitly `float32` / `float64` parameterized
- benchmark output writes the shared benchmark schema and runtime manifest
- notebook-facing stdout remains human-readable

Interpretation notes:

- `dense_plan_prepare` measures preparation overhead separately from plan reuse
- `cached_matvec_padded` exists to exercise the compile-stable service pattern, not to claim best single-shot latency
- structured SPD / HPD plan solves should be interpreted separately from general LU/direct solves
- block assembly timings are storage/layout overhead, not linear-solve quality

Current engineering readout:

- dense benchmarks are present and normalized on the shared schema
- direct and plan-reuse surfaces are benchmarked for both `arb_mat` and `acb_mat`
- the remaining dense hardening work is no longer benchmark availability; it is tighter rigorous determinant and solve/inverse semantics
