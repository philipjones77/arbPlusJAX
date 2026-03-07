Last updated: 2026-03-07T00:00:00Z

# Custom Core Status

Generated from `tools/custom_core_report.py` using `tools/core_status_report.py` and `tools/point_status_report.py`.

Scope: functions that complement the Arb-style core surface with custom pi-scaled, paired-output, fused, mixed-argument, or extended special-function helpers.

Summary: `functions=44`, `point=44/44`, `basic=44/44`, `adaptive=44/44`, `rigorous_specialized=44/44`, `generic_rigorous=0/44`.

Interpretation:
- `point`: `*_point` wrapper exists
- `basic`: `*_prec` interval/box entry point exists
- `adaptive`: adaptive mode path exists
- `rigorous_specialized`: function has a dedicated rigorous adapter
- `generic_rigorous`: rigorous mode is available through generic wrapper/kernel machinery, but not via a hand-specialized adapter

## Status Table

| function | module | family | point | basic | adaptive | rigorous_specialized | generic_rigorous | tightening_priority | notes |
|---|---|---|---|---|---|---|---|---|---|
| arb_sin_pi | arb_core | real | yes | yes | yes | yes | no | P2 | pi-scaled elementary wrapper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_cos_pi | arb_core | real | yes | yes | yes | yes | no | P2 | pi-scaled elementary wrapper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_tan_pi | arb_core | real | yes | yes | yes | yes | no | P1 | pole-sensitive pi-scaled elementary wrapper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinc | arb_core | real | yes | yes | yes | yes | no | P1 | removable singularity at zero; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinc_pi | arb_core | real | yes | yes | yes | yes | no | P1 | pi-scaled removable singularity; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sign | arb_core | real | yes | yes | yes | yes | no | P2 | discontinuous helper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_pow_fmpq | arb_core | real | yes | yes | yes | yes | no | P0 | mixed rational-power helper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_root | arb_core | real | yes | yes | yes | yes | no | P0 | mixed root helper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_cbrt | arb_core | real | yes | yes | yes | yes | no | P1 | custom root specialization; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_lgamma | arb_core | real | yes | yes | yes | yes | no | P0 | special-function complement to gamma; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_rgamma | arb_core | real | yes | yes | yes | yes | no | P0 | reciprocal gamma is numerically sensitive; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| arb_sinh_cosh | arb_core | real | yes | yes | yes | yes | no | P2 | paired-output helper; interval kernel is the rigorous path; adaptive uses dedicated or generic tightening |
| acb_rsqrt | acb_core | complex | yes | yes | yes | yes | no | P1 | complex reciprocal square-root helper; specialized rigorous adapter and adaptive path present |
| acb_cot | acb_core | complex | yes | yes | yes | yes | no | P1 | pole-sensitive trigonometric complement; specialized rigorous adapter and adaptive path present |
| acb_sech | acb_core | complex | yes | yes | yes | yes | no | P2 | hyperbolic complement; specialized rigorous adapter and adaptive path present |
| acb_csch | acb_core | complex | yes | yes | yes | yes | no | P1 | pole-sensitive hyperbolic complement; specialized rigorous adapter and adaptive path present |
| acb_sin_pi | acb_core | complex | yes | yes | yes | yes | no | P2 | pi-scaled elementary wrapper; specialized rigorous adapter and adaptive path present |
| acb_cos_pi | acb_core | complex | yes | yes | yes | yes | no | P2 | pi-scaled elementary wrapper; specialized rigorous adapter and adaptive path present |
| acb_sin_cos_pi | acb_core | complex | yes | yes | yes | yes | no | P2 | paired pi-scaled helper; specialized rigorous adapter and adaptive path present |
| acb_tan_pi | acb_core | complex | yes | yes | yes | yes | no | P1 | pole-sensitive pi-scaled elementary wrapper; specialized rigorous adapter and adaptive path present |
| acb_cot_pi | acb_core | complex | yes | yes | yes | yes | no | P1 | pole-sensitive pi-scaled complement; specialized rigorous adapter and adaptive path present |
| acb_csc_pi | acb_core | complex | yes | yes | yes | yes | no | P1 | pole-sensitive pi-scaled complement; specialized rigorous adapter and adaptive path present |
| acb_sinc | acb_core | complex | yes | yes | yes | yes | no | P1 | removable singularity at zero; specialized rigorous adapter and adaptive path present |
| acb_sinc_pi | acb_core | complex | yes | yes | yes | yes | no | P1 | pi-scaled removable singularity; specialized rigorous adapter and adaptive path present |
| acb_exp_pi_i | acb_core | complex | yes | yes | yes | yes | no | P1 | unit-circle exponential helper; specialized rigorous adapter and adaptive path present |
| acb_exp_invexp | acb_core | complex | yes | yes | yes | yes | no | P1 | paired-output exponential helper; specialized rigorous adapter and adaptive path present |
| acb_addmul | acb_core | complex | yes | yes | yes | yes | no | P2 | fused arithmetic helper; specialized rigorous adapter and adaptive path present |
| acb_submul | acb_core | complex | yes | yes | yes | yes | no | P2 | fused arithmetic helper; specialized rigorous adapter and adaptive path present |
| acb_pow_arb | acb_core | complex | yes | yes | yes | yes | no | P0 | mixed complex/real power helper; specialized rigorous adapter and adaptive path present |
| acb_pow_si | acb_core | complex | yes | yes | yes | yes | no | P2 | signed integer power helper; specialized rigorous adapter and adaptive path present |
| acb_sqr | acb_core | complex | yes | yes | yes | yes | no | P2 | square specialization; specialized rigorous adapter and adaptive path present |
| acb_root_ui | acb_core | complex | yes | yes | yes | yes | no | P1 | root helper; specialized rigorous adapter and adaptive path present |
| acb_lgamma | acb_core | complex | yes | yes | yes | yes | no | P0 | special-function complement to gamma; specialized rigorous adapter and adaptive path present |
| acb_log_sin_pi | acb_core | complex | yes | yes | yes | yes | no | P0 | branch-sensitive special helper; specialized rigorous adapter and adaptive path present |
| acb_digamma | acb_core | complex | yes | yes | yes | yes | no | P0 | special derivative function; specialized rigorous adapter and adaptive path present |
| acb_zeta | acb_core | complex | yes | yes | yes | yes | no | P0 | major special function; specialized rigorous adapter and adaptive path present |
| acb_hurwitz_zeta | acb_core | complex | yes | yes | yes | yes | no | P0 | major special function with branch structure; specialized rigorous adapter and adaptive path present |
| acb_polygamma | acb_core | complex | yes | yes | yes | yes | no | P0 | higher special derivative; specialized rigorous adapter and adaptive path present |
| acb_bernoulli_poly_ui | acb_core | complex | yes | yes | yes | yes | no | P2 | polynomial helper; specialized rigorous adapter and adaptive path present |
| acb_polylog | acb_core | complex | yes | yes | yes | yes | no | P0 | branch-sensitive special function; specialized rigorous adapter and adaptive path present |
| acb_polylog_si | acb_core | complex | yes | yes | yes | yes | no | P0 | integer-order polylog helper; specialized rigorous adapter and adaptive path present |
| acb_agm | acb_core | complex | yes | yes | yes | yes | no | P1 | iterative special helper; specialized rigorous adapter and adaptive path present |
| acb_agm1 | acb_core | complex | yes | yes | yes | yes | no | P1 | shifted iterative helper; specialized rigorous adapter and adaptive path present |
| acb_agm1_cpx | acb_core | complex | yes | yes | yes | yes | no | P1 | custom complex AGM variant; specialized rigorous adapter and adaptive path present |

## Tightening Backlog

Ranked by numerical sensitivity and the current lack of specialized rigorous adapters.

| priority | function | module | why it should move first |
|---|---|---|---|
| complete | none | n/a | explicit rigorous adapters now cover the full custom-core set |

## Immediate Readout

- Coverage is complete for `point`, `basic`, `adaptive`, and explicit rigorous dispatch across this custom-core set.
- The previous tightening backlog is closed at the mode-dispatch layer.
- Remaining quality work, if needed, is method-level improvement inside specific core kernels rather than wrapper coverage.
