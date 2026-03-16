Last updated: 2026-03-07T00:00:00Z

# Core Function Status

Generated from `src/arbplusjax/arb_core.py`, `src/arbplusjax/acb_core.py`, and `src/arbplusjax/core_wrappers.py`.

Summary: `implemented=103/103`, `adaptive=103/103`, `rigorous_specialized=60/103`, `basic_only=0/103`.

Columns:
- `implemented`: public function exists in the core module
- `basic`: `*_prec` entry point exists
- `adaptive`: adaptive mode path exists through a dedicated or generic wrapper
- `rigorous_specialized`: function has a dedicated rigorous adapter in `core_wrappers.py`
- `basic_only`: implemented/basic, but no adaptive path and no specialized rigorous path
- `kernel_split`: whether the family shares dispatch while keeping separate mode kernels
- `helper_consolidation`: whether constants/dtype/helper math has been centralized in shared substrate layers
- `batch`, `ad`, `hardening`: current engineering state pulled from the repo-wide engineering policy registry

| function | module | implemented | basic | adaptive | rigorous_specialized | basic_only | kernel_split | helper_consolidation | batch | ad | hardening | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arb_abs | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_acos | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_acosh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_add | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_asin | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_asinh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_atan | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_atanh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_cbrt | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_cos | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_cos_pi | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_cosh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_div | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_exp | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_expm1 | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_fma | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_gamma | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_inv | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_lgamma | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_log | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_log1p | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_mul | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_pow | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_pow_fmpq | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_pow_fmpz | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_pow_ui | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_rgamma | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_root | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_root_ui | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sign | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sin | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_sin_cos | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sin_pi | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinc | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinc_pi | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_sinh_cosh | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sqrt | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_sub | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_tan | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| arb_tan_pi | arb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_tanh | arb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic tightening |
| acb_abs | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | specialized rigorous adapter and adaptive path present |
| acb_acos | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_acosh | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_add | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_addmul | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_agm | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_agm1 | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_agm1_cpx | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_asin | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_asinh | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_atan | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_atanh | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_bernoulli_poly_ui | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_cos | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_cos_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_cosh | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_cot | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_cot_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_csc_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_csch | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_digamma | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_div | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_exp | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_exp_invexp | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_exp_pi_i | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_expm1 | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_gamma | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | hardened | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_hurwitz_zeta | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_inv | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_lgamma | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_log | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_log1p | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_log_sin_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_mul | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_polygamma | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_polylog | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_polylog_si | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_pow | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | hardened | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_pow_arb | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_pow_fmpz | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_pow_si | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_pow_ui | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | hardened | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_rgamma | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_root_ui | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_rsqrt | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sech | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sin | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_sin_cos | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | specialized rigorous adapter and adaptive path present |
| acb_sin_cos_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sin_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sinc | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sinc_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sinh | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_sqr | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_sqrt | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_sub | acb_core | yes | yes | yes | no | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | minimal | adaptive uses generic wrapper or dedicated adapter; rigorous uses generic wrapper |
| acb_submul | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_tan | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_tan_pi | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
| acb_tanh | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | generic_or_mixed | specialized rigorous adapter and adaptive path present |
| acb_zeta | acb_core | yes | yes | yes | yes | no | shared_dispatch_separate_mode_kernels | shared_elementary_or_core | mixed | mixed | specialized | specialized rigorous adapter and adaptive path present |
