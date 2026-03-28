# Example API Surface Summary (cpu)

- python: `/home/phili/miniforge3/envs/jax/bin/python`
- backend: `cpu`
- api_rows: `8`
- diagnostics_rows: `8`

## Routed Operations

- `acb_mat_solve` / `direct`: warm=1.03997e-05s, cold=0.206161s, recompile=1.0394e-05s
- `acb_mat_solve` / `routed`: warm=1.2917e-05s, cold=0.144443s, recompile=1.1234e-05s
- `arb_mat_solve` / `direct`: warm=1.37173e-05s, cold=0.412853s, recompile=1.2744e-05s
- `arb_mat_solve` / `routed`: warm=1.19683e-05s, cold=0.118087s, recompile=9.526e-06s
- `besselk` / `direct_cuda_besselk`: warm=8.76833e-06s, cold=0.149198s, recompile=8.554e-06s
- `besselk` / `routed_cuda_besselk`: warm=9.777e-06s, cold=0.358417s, recompile=1.0508e-05s
- `incomplete_gamma_upper` / `direct`: warm=1.9013e-05s, cold=0.150323s, recompile=2.0815e-05s
- `incomplete_gamma_upper` / `routed`: warm=1.46887e-05s, cold=0.0984886s, recompile=1.175e-05s

## Diagnostics Cases

- `arb_dense_matvec_cached_apply`: compile_ms=116.403, steady_ms_median=0.291932, recompile_new_shape_ms=113.605
- `acb_dense_matvec_cached_apply`: compile_ms=167.589, steady_ms_median=0.226531, recompile_new_shape_ms=169.808
- `srb_sparse_matvec_point`: compile_ms=37.7179, steady_ms_median=0.046928, recompile_new_shape_ms=33.9338
- `jrb_operator_apply_point`: compile_ms=49.8786, steady_ms_median=0.059877, recompile_new_shape_ms=57.4105
- `jrb_logdet_slq_point`: compile_ms=359.343, steady_ms_median=0.194831, recompile_new_shape_ms=217.258
- `jcb_operator_apply_point`: compile_ms=60.5496, steady_ms_median=0.042202, recompile_new_shape_ms=63.3422
- `jcb_logdet_slq_point`: compile_ms=363.474, steady_ms_median=0.247365, recompile_new_shape_ms=297.592
- `jcb_sparse_logdet_slq_point`: compile_ms=262.773, steady_ms_median=0.13427, recompile_new_shape_ms=0.115539
