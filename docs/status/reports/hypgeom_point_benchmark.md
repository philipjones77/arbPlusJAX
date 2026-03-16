Last updated: 2026-03-08T00:00:00Z

# Hypgeom Point Benchmark

CPU timing smoke benchmark at batch size `128`.

| mode | function | mean_time_ms |
|---|---|---:|
| point | `arb_hypgeom_1f1` | 24.5192 |
| point | `boost_hypergeometric_1f1` | 29.9140 |
| point | `arb_hypgeom_2f1` | 24.3269 |
| point | `boost_hyp2f1_series` | 38.3142 |
| point | `arb_hypgeom_u` | 2.5013 |
| basic | `arb_hypgeom_1f1` | 16.5735 |
| basic | `boost_hypergeometric_1f1` | 17.3505 |
| basic | `arb_hypgeom_2f1` | 19.3014 |
| basic | `boost_hyp2f1_series` | 20.9791 |
| basic | `arb_hypgeom_pfq` | 0.1946 |
| basic | `boost_hypergeometric_pfq` | 0.2533 |
