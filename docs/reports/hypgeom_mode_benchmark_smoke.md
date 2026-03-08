Last updated: 2026-03-07T00:00:00Z

# Hypgeom Mode Benchmark Smoke

Small CPU smoke benchmark for selected canonical and alternative hypgeom families.

Comparisons:
- `unpadded_basic_ms`: current batch path
- `padded_basic_ms`: fixed-shape padded batch path

| family | path | unpadded_basic_ms | padded_basic_ms |
|---|---|---:|---:|
| arb_hypgeom_gamma | api.eval_interval_batch | 0.8521 | 0.6204 |
| arb_hypgeom_erf | api.eval_interval_batch | 0.1716 | 0.7815 |
| arb_hypgeom_ei | api.eval_interval_batch | 0.2614 | 0.8466 |
| arb_hypgeom_gamma_lower | api.eval_interval_batch | 1.1188 | 4.3621 |
| arb_hypgeom_gamma_upper | api.eval_interval_batch | 2.3486 | 1.5822 |
| arb_hypgeom_0f1 | api.eval_interval_batch | 0.3617 | 3.7904 |
| arb_hypgeom_1f1 | api.eval_interval_batch | 7.1998 | 11.2229 |
| arb_hypgeom_2f1 | api.eval_interval_batch | 5.1005 | 5.4083 |
| arb_hypgeom_u | api.eval_interval_batch | 1.9709 | 3.6842 |
| arb_hypgeom_pfq | api.eval_interval_batch | 0.5538 | 1.5763 |
| boost_hypergeometric_1f1 | api.eval_interval_batch | 3.7474 | 5.8278 |
| cusf_hyp1f1 | api.eval_interval_batch | 3.0399 | 8.1240 |
| arb_hypgeom_legendre_p | api.eval_interval_batch | 0.5186 | 3.0985 |
| arb_hypgeom_jacobi_p | api.eval_interval_batch | 0.9881 | 2.2462 |
| arb_hypgeom_gegenbauer_c | api.eval_interval_batch | 1.1215 | 4.5920 |
| arb_hypgeom_chebyshev_t | api.eval_interval_batch | 0.3676 | 1.7334 |
| arb_hypgeom_chebyshev_u | api.eval_interval_batch | 0.6981 | 2.2016 |
| arb_hypgeom_laguerre_l | api.eval_interval_batch | 0.5751 | 1.3026 |
| arb_hypgeom_hermite_h | api.eval_interval_batch | 0.3271 | 2.2961 |
