# Example Run Suite Summary: cpu_stage1_smoke

## Profile backends

| backend | mean_time_ms | mean_containment | rows |
|---|---:|---:|---:|
| boost | 12.1655 | 1 | 2 |
| c_arb | 0.0458415 | 0 | 2 |
| jax_adaptive | 0.457475 | 0 | 2 |
| jax_basic | 0.389629 | 0 | 2 |
| jax_point | 0.43592 | 1 | 2 |
| jax_rigorous | 0.390352 | 0 | 2 |
| mpmath | 0.464985 | 1 | 2 |
| scipy | 0.0115905 | 1 | 2 |

## API benchmark summary

| operation | implementation | cold_time_s | warm_time_s | recompile_time_s |
|---|---|---:|---:|---:|
| besselk | direct_cuda_besselk | 0.0833113 | 1.5091e-05 | 9.768e-06 |
| besselk | routed_cuda_besselk | 0.38133 | 1.1702e-05 | 9.651e-06 |
| incomplete_gamma_upper | direct | 0.155665 | 1.3027e-05 | 1.1637e-05 |
| incomplete_gamma_upper | routed | 0.0973854 | 1.2457e-05 | 1.157e-05 |
| arb_mat_solve | direct | 0.338992 | 1.9613e-05 | 1.6465e-05 |
| arb_mat_solve | routed | 0.105712 | 1.4907e-05 | 1.2214e-05 |
| acb_mat_solve | direct | 0.183527 | 1.4659e-05 | 1.0285e-05 |
| acb_mat_solve | routed | 0.138229 | 1.7827e-05 | 1.4969e-05 |

## Matrix diagnostics summary

| case | compile_ms | steady_ms_median | recompile_new_shape_ms | peak_rss_delta_mb |
|---|---:|---:|---:|---:|
| arb_dense_matvec_cached_apply | 121.359 | 0.042077 | 110.622 | 14.6875 |
| acb_dense_matvec_cached_apply | 217.505 | 0.075028 | 198.804 | 19.2188 |
| srb_sparse_matvec_point | 42.8409 | 0.054642 | 37.0558 | 0 |
| jrb_operator_apply_point | 55.9705 | 0.062971 | 56.5849 | 1.5625 |
| jrb_logdet_slq_point | 444.505 | 0.111046 | 210.211 | 45.1367 |
| jcb_operator_apply_point | 107.582 | 0.054814 | 83.959 | 1.25 |
| jcb_logdet_slq_point | 462.843 | 0.266677 | 432.575 | 21.25 |
| jcb_sparse_logdet_slq_point | 488.841 | 0.218386 | 0.170195 | 10.4688 |
