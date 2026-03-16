Last updated: 2026-03-07T00:00:00Z

# Hypgeom Mode Benchmark Smoke

Small CPU smoke benchmark for selected canonical and alternative hypgeom families.

Comparisons:
- `unpadded_basic_ms`: current batch path
- `padded_basic_ms`: fixed-shape padded batch path

| family | path | unpadded_basic_ms | padded_basic_ms |
|---|---|---:|---:|
| arb_hypgeom_gamma | api.eval_interval_batch | 0.2496 | 1.4617 |
| arb_hypgeom_erf | api.eval_interval_batch | 0.3084 | 2.7260 |
| arb_hypgeom_ei | api.eval_interval_batch | 0.3369 | 0.9348 |
| arb_hypgeom_gamma_lower | api.eval_interval_batch | 0.5522 | 2.9553 |
| arb_hypgeom_gamma_upper | api.eval_interval_batch | 0.7530 | 3.5797 |
| arb_hypgeom_0f1 | api.eval_interval_batch | 1.0877 | 2.7949 |
| arb_hypgeom_1f1 | api.eval_interval_batch | 15.8036 | 13.1651 |
| arb_hypgeom_2f1 | api.eval_interval_batch | 4.1174 | 4.8614 |
| arb_hypgeom_u | api.eval_interval_batch | 0.9454 | 3.5892 |
| arb_hypgeom_pfq | api.eval_interval_batch | 0.2755 | 1.9731 |
| boost_hypergeometric_1f1 | api.eval_interval_batch | 2.8295 | 6.0154 |
| cusf_hyp1f1 | api.eval_interval_batch | 3.1367 | 5.1604 |
| arb_hypgeom_legendre_p | api.eval_interval_batch | 0.6510 | 1.2264 |
| arb_hypgeom_jacobi_p | api.eval_interval_batch | 0.5626 | 1.3308 |
| arb_hypgeom_gegenbauer_c | api.eval_interval_batch | 0.4459 | 1.5504 |
| arb_hypgeom_chebyshev_t | api.eval_interval_batch | 0.2350 | 1.6265 |
| arb_hypgeom_chebyshev_u | api.eval_interval_batch | 0.7263 | 1.7291 |
| arb_hypgeom_laguerre_l | api.eval_interval_batch | 0.2821 | 1.0903 |
| arb_hypgeom_hermite_h | api.eval_interval_batch | 0.3554 | 1.6534 |
