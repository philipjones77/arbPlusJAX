# Dense Matrix Surface Benchmark

Date: 2026-03-18
Status: active

## Scope

This report covers the pure-JAX dense `arb_mat` / `acb_mat` matrix surface benchmark in:

- [benchmark_dense_matrix_surface.py](/home/phili/projects/arbplusJAX/benchmarks/benchmark_dense_matrix_surface.py)

It focuses on:

- direct solve
- LU-reuse solve
- LU-reuse transpose solve and add-solve
- dense arithmetic (`add`, entrywise multiply)
- dense matrix functions (`charpoly`, `pow_ui`, `exp`)
- triangular and LU-style solve aliases
- SPD / HPD structured solve
- SPD / HPD structured plan-reuse solve
- SPD / HPD structured eigendecomposition
- dense matvec plan preparation
- cached matvec
- cached matvec padded batch apply
- transpose / conjugate-transpose
- diagonal extraction

## Environment

- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- backend: `cpu`
- GPU note: CUDA-enabled `jaxlib` was not installed, so JAX ran on CPU

## Results

### `n=16`, `warmup=2`, `runs=3`

- `arb_direct_solve_s`: `0.000033`
- `arb_lu_reuse_s`: `0.000050`
- `arb_lu_reuse_transpose_s`: `0.000072`
- `arb_lu_reuse_add_s`: `0.000091`
- `arb_dense_plan_prepare_s`: `0.000022`
- `arb_cached_matvec_s`: `0.000020`
- `arb_cached_matvec_padded_s`: `0.005764`
- `arb_add_s`: `0.000008`
- `arb_mul_entrywise_s`: `0.000087`
- `arb_charpoly_s`: `0.000167`
- `arb_pow_ui_s`: `0.000016`
- `arb_exp_s`: `0.000110`
- `arb_spd_solve_s`: `0.000134`
- `arb_spd_plan_solve_s`: `0.000122`
- `arb_spd_eigh_s`: `0.000196`
- `arb_solve_tril_s`: `0.000013`
- `arb_solve_lu_s`: `0.000122`
- `arb_transpose_s`: `0.000007`
- `arb_diag_s`: `0.000007`
- `arb_block_assemble_s`: `0.000322`
- `arb_block_diag_s`: `0.001762`
- `acb_direct_solve_s`: `0.000175`
- `acb_lu_reuse_s`: `0.000151`
- `acb_lu_reuse_transpose_s`: `0.000105`
- `acb_lu_reuse_add_s`: `0.000060`
- `acb_dense_plan_prepare_s`: `0.000017`
- `acb_cached_matvec_s`: `0.000112`
- `acb_cached_matvec_padded_s`: `0.023397`
- `acb_add_s`: `0.000035`
- `acb_mul_entrywise_s`: `0.000187`
- `acb_charpoly_s`: `0.000297`
- `acb_pow_ui_s`: `0.000028`
- `acb_exp_s`: `0.000130`
- `acb_hpd_solve_s`: `0.000052`
- `acb_hpd_plan_solve_s`: `0.000031`
- `acb_hpd_eigh_s`: `0.000127`
- `acb_solve_tril_s`: `0.000075`
- `acb_solve_lu_s`: `0.000079`
- `acb_transpose_s`: `0.000013`
- `acb_conjugate_transpose_s`: `0.000014`
- `acb_diag_s`: `0.000008`
- `acb_block_assemble_s`: `0.000166`
- `acb_block_diag_s`: `0.004736`

## Notes

- Dense plan preparation itself is cheap on CPU at this size; the dominant extra cost is padded batch apply, not plan construction.
- The padded cached matvec path is materially slower than single-vector cached apply at `n=16`, which is expected because it trades raw latency for fixed-shape batching and reduced recompilation pressure.
- The structured dense paths are now competitive with the generic direct solve and cheaper once a Cholesky plan is reused.
- Structured eigendecomposition is now benchmarked alongside structured solve so the dense symmetric/Hermitian spectral path has the same validation loop as the factorization path.
- The broader dense surface now has measured arithmetic and solve-alias timings, so `add`, entrywise multiply, `solve_tril`, and `solve_lu` no longer sit outside the benchmark loop.
- The PETSc-style dense factorization-solve tranche is now in the loop as well, so transpose solve and add-solve on top of cached LU factors regress with the rest of the dense surface.
- The dense matrix-function tranche is now benchmarked as well, so `charpoly`, `pow_ui`, and `exp` are part of the same regression loop as the solve paths.
- The latest dense tranche correctness sweep for the related dense tests finished at `55 passed in 82.23s`.
- The current rigorous dense matrix layer is still wrapper/enclosure based for solve-like operations, but exact structural transforms now preserve interval/box information directly for:
  - permutation matrices
  - transpose / conjugate-transpose
  - symmetric / Hermitian part extraction
  - submatrix extraction
  - diagonal extraction
  - diagonal-matrix construction
  - dense matvec plan preparation
  - dense LU-reuse plan preparation
  - dense SPD / HPD plan preparation

## Related Surfaces

- example notebook: [example_dense_matrix_surface.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_matrix_surface.ipynb)
- structured spectral example notebook: [example_dense_structured_spectral.ipynb](/home/phili/projects/arbplusJAX/examples/example_dense_structured_spectral.ipynb)
- dense matrix tests:
  - [test_arb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_arb_mat_chassis.py)
  - [test_acb_mat_chassis.py](/home/phili/projects/arbplusJAX/tests/test_acb_mat_chassis.py)
  - [test_mat_modes.py](/home/phili/projects/arbplusJAX/tests/test_mat_modes.py)
  - [test_dense_plan_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_plan_modes.py)
  - [test_dense_structured_modes.py](/home/phili/projects/arbplusJAX/tests/test_dense_structured_modes.py)
  - [test_dense_eigh_and_aliases.py](/home/phili/projects/arbplusJAX/tests/test_dense_eigh_and_aliases.py)
