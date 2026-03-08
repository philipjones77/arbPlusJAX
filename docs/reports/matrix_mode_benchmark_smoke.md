Last updated: 2026-03-07T00:00:00Z

# Matrix Mode Benchmark Smoke

Quick warmed benchmark for canonical `arb_mat` / `acb_mat` `n x n` determinant and trace paths.

| function | mode | mean_time_ms |
|---|---:|---:|
| arb_mat_det | point | 0.0167 |
| arb_mat_det | basic | 57.4621 |
| arb_mat_det | rigorous | 50.2211 |
| arb_mat_trace | point | 0.0204 |
| arb_mat_trace | basic | 4.5220 |
| arb_mat_trace | rigorous | 5.4453 |
| acb_mat_det | point | 0.0091 |
| acb_mat_det | basic | 256.6520 |
| acb_mat_det | rigorous | 320.9762 |
| acb_mat_trace | point | 0.0115 |
| acb_mat_trace | basic | 8.9144 |
| acb_mat_trace | rigorous | 11.1021 |
