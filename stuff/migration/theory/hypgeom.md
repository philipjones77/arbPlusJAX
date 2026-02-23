# hypgeom

## Scope

- Migration scaffolding for a focused hypergeometric subset used by the C ref library.
- Naive series/elementary implementations evaluated at interval midpoints, with outward rounding.

## Intended API Surface

- C reference library: `hypgeom_ref`
  - `arb_hypgeom_rising_ui_forward_ref`, `arb_hypgeom_rising_ui_ref`
  - `arb_hypgeom_lgamma_ref`, `arb_hypgeom_gamma_ref`, `arb_hypgeom_rgamma_ref`
  - `arb_hypgeom_erf_ref`, `arb_hypgeom_erfc_ref`, `arb_hypgeom_erfi_ref`
  - `arb_hypgeom_erfinv_ref`, `arb_hypgeom_erfcinv_ref`
  - `arb_hypgeom_0f1_ref`, `arb_hypgeom_m_ref`, `arb_hypgeom_1f1_ref`, `arb_hypgeom_1f1_full_ref`
  - `arb_hypgeom_2f1_ref`, `arb_hypgeom_2f1_full_ref`
  - `arb_hypgeom_u_ref`
  - `arb_hypgeom_bessel_j_ref`, `arb_hypgeom_bessel_y_ref`, `arb_hypgeom_bessel_jy_ref`
  - `arb_hypgeom_bessel_i_ref`, `arb_hypgeom_bessel_k_ref`
  - Scaled and integration wrappers
  - Batch variants
- JAX module: `arbjax.hypgeom`
  - Matching `arb_hypgeom_*` and `acb_hypgeom_*` APIs
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Series evaluated at real/complex midpoints only.
- Outward rounding applied to final interval or complex box.
- If non-finite, return a full interval/box.

## Differentiability

- Differentiable w.r.t. midpoint parameters for smooth subdomains.
- Truncation counts and mode switches are static JIT args.

## Notes

- Implemented extra real special functions: Fresnel, Ei, Si, Ci, Shi, Chi, li, dilog (midpoint + outward rounding).
- Si/Ci use a small-|x| series; values outside |x|<=4 return full intervals.
- Implemented additional real special functions: Airy (Ai/Bi + derivatives), expint (integer order only), incomplete gamma (lower/upper), incomplete beta, Chebyshev, Laguerre, Hermite, Legendre P/Q (m=0 only), Jacobi, Gegenbauer, central binomial.
- Added implementations for the remaining Arb APIs using midpoint Taylor series or simple asymptotics where available.
- Some advanced helpers (gamma Stirling sum helpers, Coulomb) use very rough placeholders and should be treated as scaffolding only.
## Benchmarks

- JAX-only benchmarks: `migration/tools/benchmark_hypgeom_extra.py`
- This is not a full Arb replacement; it covers the hypergeometric subset in `hypgeom_ref` plus these extras.
- Bessel functions are series-based; accuracy degrades for large order/argument.
