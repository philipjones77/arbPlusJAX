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
- JAX module: `arbplusjax.hypgeom`
  - Matching `arb_hypgeom_*` and `acb_hypgeom_*` APIs
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Baseline: series evaluated at real/complex midpoints only, outward rounded.
- Rigorous/adaptive: special functions (erf/erfc/erfi, erfinv/erfcinv, Ei/Si/Ci/Shi/Chi/Li, dilog, Fresnel, Airy, gamma, Bessel J/Y/I/K and scaled variants) use ball‑style Lipschitz/sampling bounds.
- 0F1 / 1F1 / 2F1 / U use explicit series remainder bounds in rigorous mode (ratio‑test tail bound).
- Remaining functions use Jacobian/sampling bounds from `hypgeom_wrappers` until analytic interval formulas are added.
- Legendre / Jacobi / Gegenbauer helpers now use interval recurrences or interval coefficient sums in rigorous mode.

## Differentiability

- Differentiable w.r.t. midpoint parameters for smooth subdomains.
- Truncation counts and mode switches are static JIT args.

## Notes

- Implemented extra real special functions: Fresnel, Ei, Si, Ci, Shi, Chi, li, dilog (midpoint + outward rounding).
- Si/Ci use a small-|x| series; values outside |x|<=4 return full intervals.
- Implemented additional real special functions: Airy (Ai/Bi + derivatives), expint (integer order only), incomplete gamma (lower/upper), incomplete beta, Chebyshev, Laguerre, Hermite, Legendre P/Q (m=0 only), Jacobi, Gegenbauer, central binomial.
- Added implementations for the remaining Arb APIs using midpoint Taylor series or simple asymptotics where available.
-- Stirling sum helpers now use a finite Stirling series with a conservative tail bound.
-- Coulomb and certain jet/series helpers still use midpoint Taylor coefficients with conservative coefficient inflation.
## Benchmarks

- JAX-only benchmarks: `tools/benchmark_hypgeom_extra.py`
- This is not a full Arb replacement; it covers the hypergeometric subset in `hypgeom_ref` plus these extras.
- Bessel functions are series-based; accuracy degrades for large order/argument.

## Formulas

- Γ(z), log Γ(z), 1/Γ(z) from Lanczos + reflection.
- erf/erfc/erfi via series/exp relations on midpoint.
- 0F1, 1F1, 2F1 via truncated hypergeometric series.
- Bessel J/I via power series; Y/K from J/I combinations.
- Fresnel S,C via definition with normalization.
- Ei, Si, Ci, Shi, Chi via series or expi relations.
- Incomplete gamma/beta via `gammainc/gammaincc/betainc` on midpoint.
- Orthogonal polynomials via explicit sums/recurrences.

## Implementation Notes

- Most functions are midpoint-evaluated with outward rounding.
- Some helper/jet/series APIs use Taylor coefficients from autodiff.


## Barnes G

We implement Barnes G using a log-asymptotic expansion for `log G(z)` with the recurrence `G(z+1)=Γ(z)G(z)`.
Baseline uses midpoint evaluation and endpoint sampling; adaptive/rigorous wrappers use Lipschitz/gradient bounds over intervals.
