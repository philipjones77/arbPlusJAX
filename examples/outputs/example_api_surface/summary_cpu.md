# Example API Surface Summary (cpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `cpu`
- api_rows: `8`
- diagnostics_rows: `8`

## Routed Operations

- `acb_mat_solve` / `direct`: warm=1.90687e-05s, cold=0.224302s, recompile=1.7988e-05s
- `acb_mat_solve` / `routed`: warm=1.98103e-05s, cold=0.174009s, recompile=1.2638e-05s
- `arb_mat_solve` / `direct`: warm=1.76437e-05s, cold=0.407904s, recompile=1.6739e-05s
- `arb_mat_solve` / `routed`: warm=1.2266e-05s, cold=0.167525s, recompile=1.124e-05s
- `besselk` / `direct_cuda_besselk`: warm=2.3523e-05s, cold=0.102939s, recompile=1.0867e-05s
- `besselk` / `routed_cuda_besselk`: warm=2.12667e-05s, cold=0.538709s, recompile=1.0175e-05s
- `incomplete_gamma_upper` / `direct`: warm=1.82173e-05s, cold=0.137871s, recompile=1.3549e-05s
- `incomplete_gamma_upper` / `routed`: warm=1.81363e-05s, cold=0.12041s, recompile=1.3374e-05s

## Diagnostics Cases

- `arb_dense_matvec_cached_apply`: compile_ms=128.933, steady_ms_median=0.099928, recompile_new_shape_ms=140.769
- `acb_dense_matvec_cached_apply`: compile_ms=194.618, steady_ms_median=0.127457, recompile_new_shape_ms=196.677
- `srb_sparse_matvec_point`: compile_ms=43.4593, steady_ms_median=0.074519, recompile_new_shape_ms=46.4703
- `jrb_operator_apply_point`: compile_ms=68.281, steady_ms_median=0.059541, recompile_new_shape_ms=71.0329
- `jrb_logdet_slq_point`: compile_ms=512.159, steady_ms_median=0.427156, recompile_new_shape_ms=267.901
- `jcb_operator_apply_point`: compile_ms=87.9586, steady_ms_median=0.075568, recompile_new_shape_ms=85.9828
- `jcb_logdet_slq_point`: compile_ms=423.825, steady_ms_median=0.370462, recompile_new_shape_ms=358.732
- `jcb_sparse_logdet_slq_point`: compile_ms=353.165, steady_ms_median=0.312364, recompile_new_shape_ms=0.234577
