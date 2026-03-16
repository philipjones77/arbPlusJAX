Last updated: 2026-03-07T00:00:00Z

# Core Mode Benchmark Smoke

CPU smoke benchmark with warmed JIT kernels (`JAX_PLATFORMS=cpu`, `prec_bits=80`, 200 iterations per case).

## Real core

| function | basic_ms | rigorous_ms | adaptive_ms | basic_width | rigorous_width | adaptive_width |
|---|---:|---:|---:|---:|---:|---:|
| arb_asin | 0.024893 | 0.010321 | 0.016949 | 0.103335 | 0.103335 | 0.104025 |
| arb_log1p | 0.007797 | 0.013799 | 0.018014 | 0.0800427 | 0.0800427 | 0.081644 |
| arb_gamma | 0.028002 | 0.019686 | 0.058828 | 0.0309049 | 0.0309049 | 0.0413961 |
| arb_pow_ui | 0.021810 | 0.025012 | 0.045439 | 1.352 | 1.352 | 1.442 |

## Complex core

| function | basic_ms | rigorous_ms | adaptive_ms | basic_re_width | rigorous_re_width | adaptive_re_width | basic_im_width | rigorous_im_width | adaptive_im_width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acb_asin | 0.022603 | 0.057424 | 0.049883 | 4.44089e-16 | 0.10328 | 0.104753 | 0 | 0.206559 | 0.209163 |
| acb_log1p | 0.017006 | 0.016093 | 0.043291 | 4.44089e-16 | 0.08 | 0.0843411 | 0 | 0.16 | 0.166282 |
| acb_gamma | 0.067511 | 0.124617 | 0.365994 | 0.0349838 | 0.035692 | 0.0363547 | 0.0350732 | 0.0376642 | 0.0350732 |
| acb_pow_ui | 0.009870 | 0.009295 | 0.015373 | 5.32907e-15 | 0.7545 | 0.4975 | 1.77636e-15 | 0.7545 | 0.7985 |
