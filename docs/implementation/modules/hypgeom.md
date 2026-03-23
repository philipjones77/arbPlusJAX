Last updated: 2026-02-25T03:51:38Z

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

- basic: series evaluated at real/complex midpoints only, outward rounded.
- Rigorous/adaptive: special functions (erf/erfc/erfi, erfinv/erfcinv, Ei/Si/Ci/Shi/Chi/Li, dilog, Fresnel, Airy, gamma, Bessel J/Y/I/K and scaled variants) use ball‑style Lipschitz/sampling bounds.
- 0F1 / 1F1 / 2F1 / U use explicit series remainder bounds in rigorous mode (ratio‑test tail bound).
- Remaining functions use Jacobian/sampling bounds from `hypgeom_wrappers` until analytic interval formulas are added.
- Legendre / Jacobi / Gegenbauer helpers now use interval recurrences or interval coefficient sums in rigorous mode.

## Differentiability

- Differentiable w.r.t. midpoint parameters for smooth subdomains.
- Truncation counts and mode switches are static JIT args.
- Current staged status and smoke benchmark:
  - [hypgeom_status.md](/docs/status/reports/hypgeom_status.md)
  - [hypgeom_mode_benchmark_smoke.md](/docs/status/reports/hypgeom_mode_benchmark_smoke.md)

## Notes

- Implemented extra real special functions: Fresnel, Ei, Si, Ci, Shi, Chi, li, dilog (midpoint + outward rounding).
- Si/Ci use a small-|x| series; values outside |x|<=4 return full intervals.
- Implemented additional real special functions: Airy (Ai/Bi + derivatives), expint (integer order only), incomplete gamma (lower/upper), incomplete beta, Chebyshev, Laguerre, Hermite, Legendre P/Q (m=0 only), Jacobi, Gegenbauer, central binomial.
- Added implementations for the remaining Arb APIs using midpoint Taylor series or simple asymptotics where available.
-- Stirling sum helpers now use a finite Stirling series with a conservative tail bound.
-- Coulomb and certain jet/series helpers still use midpoint Taylor coefficients with conservative coefficient inflation.
## Benchmarks

- JAX-only benchmarks: `benchmarks/benchmark_hypgeom_extra.py`
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
basic uses midpoint evaluation and endpoint sampling; adaptive/rigorous wrappers use Lipschitz/gradient bounds over intervals.

## Per-Function Documentation Template

Use this structure for each function (fill in specifics):

```
Function: <name>
Inputs: <signature, domain>
C Arb: <ball method, precision p, any special casing>
JAX basic: <midpoint eval + outward rounding, kernel>
JAX rigorous: <analytic bound / Jacobian / series remainder, key constants>
JAX adaptive: <sampling stencil, any heuristics>
SciPy/JAX SciPy: <point evaluation, dtype>
mpmath: <precision model, any series/continued fraction>
Formulas: <series/recurrence/identity>
Constants: <pi, EulerGamma, zeta, Bernoulli, etc>
Branch cuts / singularities: <how handled>
Tightening: <what and where in code>
Code references: <file paths and key functions>
Notes: <limits, accuracy regimes>
```

## Per-Function Notes (Hypgeom start)

### arb_hypgeom_gamma / acb_hypgeom_gamma

- Inputs: real interval / complex box.
- C Arb: arbitrary-precision ball arithmetic for gamma and log-gamma.
- JAX basic: Lanczos midpoint + outward rounding.
- JAX rigorous: ball wrappers with derivative/Lipschitz bounds.
- JAX adaptive: fixed sampling stencil around midpoint.
- SciPy/JAX SciPy: point gamma evaluation (float64).
- mpmath: arbitrary precision gamma; interval only with `mp.iv`.
- Formulas: Lanczos + reflection + recurrence for shifting.
- Constants: Lanczos coefficients, pi.
- Branch cuts: negative real axis (reflection).
- Tightening: uses regularized lower/upper gamma complementary relation for stability.
- Code: `src/arbplusjax/hypgeom.py` (`arb_hypgeom_gamma_prec`), `src/arbplusjax/ball_wrappers.py` (`arb_ball_gamma`, adaptive).

### arb_hypgeom_0f1 / acb_hypgeom_0f1

- Inputs: parameters + z (real/complex).
- C Arb: ball arithmetic with series + binary splitting.
- JAX basic: truncated series at midpoint with outward rounding.
- JAX rigorous: explicit tail bound (ratio-test) for series remainder.
- JAX adaptive: sampling around midpoint for bound inflation.
- SciPy/JAX SciPy: point evaluation (where available).
- mpmath: arbitrary precision hypergeometric.
- Formulas: 0F1 series.
- Constants: factorials, rising factorials.
- Branch cuts: none (entire in z, for fixed parameters).
- Tightening: explicit remainder bound in `hypgeom.py`.
- Batch/JAX: dedicated fixed-shape padded batch entry points now exist for basic/adaptive/rigorous on the canonical real and complex families.
- Code: `src/arbplusjax/hypgeom.py` (`arb_hypgeom_0f1`, `acb_hypgeom_0f1`, tail bound helpers).

### arb_hypgeom_1f1 / acb_hypgeom_1f1

- Inputs: parameters + z.
- C Arb: ball arithmetic with series and asymptotic switching.
- JAX basic: truncated series at midpoint.
- JAX rigorous: explicit tail bound; corner sampling on complex boxes.
- JAX adaptive: sampling stencil.
- SciPy/JAX SciPy: point evaluation (SciPy special).
- mpmath: arbitrary precision hypergeometric.
- Formulas: 1F1 series.
- Constants: factorials, rising factorials.
- Branch cuts: none (entire in z).
- Tightening: tail bound + explicit regime candidates (Kummer-style transform on the canonical scalar path) + corner/endpoint sampling, now assembled through shared helper builders in `hypgeom.py`.
- Batch/JAX: dedicated fixed-shape padded batch entry points now exist for basic/adaptive/rigorous on the canonical real and complex families; `m` follows the same implementation path, and the family mode-batch cores now dispatch directly to the specialized family kernels instead of generic `*_mode` lambdas.
- AD: boundary sweeps now cover the main real regime switch around `z≈6` plus deeper representative complex cut/corner-adjacent points.
- Compile behavior: targeted probe for `1f1/2f1/u` shows family compile events reduced from `6` to `3`, though total padded compile count still rises from `27` to `31` because caller/helper overhead remains.
- Code: `src/arbplusjax/hypgeom.py` (`arb_hypgeom_1f1`, `acb_hypgeom_1f1`, tail bound helpers).

### arb_hypgeom_2f1 / acb_hypgeom_2f1

- Inputs: parameters + z.
- C Arb: ball arithmetic with transformations + analytic continuation.
- JAX basic: truncated series at midpoint (|z| small), otherwise conservative.
- JAX rigorous: tail bound + endpoint/corner sampling.
- JAX adaptive: sampling stencil.
- SciPy/JAX SciPy: point evaluation (SciPy special).
- mpmath: arbitrary precision hypergeometric with analytic continuation.
- Formulas: 2F1 series in |z|<1; analytic continuation not fully mirrored.
- Constants: factorials, rising factorials.
- Branch cuts: [1, ∞) on real axis; widened bounds if box crosses cut.
- Tightening: tail bounds + explicit transform candidates + corner sampling; still no full analytic continuation. Candidate assembly is now shared through helper builders in `hypgeom.py`.
- Batch/JAX: dedicated fixed-shape padded batch entry points now exist for basic/adaptive/rigorous on the canonical real and complex families, and the family mode-batch cores now dispatch directly to the specialized family kernels instead of generic `*_mode` lambdas.
- AD: boundary sweeps now cover the main real transform switch around `|z|≈0.75` plus deeper representative complex cut/corner-adjacent points.
- Compile behavior: targeted probe for `1f1/2f1/u` shows family compile events reduced from `6` to `3`, though total padded compile count still rises from `27` to `31` because caller/helper overhead remains.
- Code: `src/arbplusjax/hypgeom.py` (`arb_hypgeom_2f1`, `acb_hypgeom_2f1`).

### arb_hypgeom_u / acb_hypgeom_u

- Inputs: parameters + z.
- C Arb: ball arithmetic with regime selection.
- JAX basic: midpoint eval, simplified regimes.
- JAX rigorous: regime selection + corner sampling.
- JAX adaptive: sampling stencil.
- SciPy/JAX SciPy: point evaluation (scipy.special.hyperu).
- mpmath: arbitrary precision hyperu.
- Formulas: series/asymptotic regimes.
- Constants: pi, gamma.
- Branch cuts: negative real axis; widened bounds if box crosses cut.
- Tightening: regime-specific bounds, explicit asymptotic candidates, and corner sampling, now assembled through shared helper builders in `hypgeom.py`.
- Batch/JAX: dedicated fixed-shape padded batch entry points now exist for basic/adaptive/rigorous on the canonical real and complex families, and the family mode-batch cores now dispatch directly to the specialized family kernels instead of generic `*_mode` lambdas; full AD guarantees across all regime switches are still a staged target.
- AD: boundary sweeps now cover the main real regime change around `z≈8` plus deeper representative complex cut/corner-adjacent points.
- Compile behavior: targeted probe for `1f1/2f1/u` shows family compile events reduced from `6` to `3`, though total padded compile count still rises from `27` to `31` because caller/helper overhead remains.
- Code: `src/arbplusjax/hypgeom.py` (`arb_hypgeom_u`, `_real_hypu_regime`, `_complex_hypu_regime`).

### Stronger orthogonal families

- Families: `arb_hypgeom_legendre_p`, `arb_hypgeom_legendre_q`, `arb_hypgeom_jacobi_p`, `arb_hypgeom_gegenbauer_c`.
- JAX basic: recurrence or coefficient evaluation at midpoint with outward rounding.
- JAX rigorous: explicit recurrence/interval kernels; `jacobi_p_rigorous` is now structured with `lax.fori_loop` instead of a Python accumulation loop.
- JAX adaptive: currently wrapper-level adaptive inflation on top of the canonical kernels, but the staged batch path now has dedicated fixed-shape entry points.
- Batch/JAX: dedicated fixed-shape padded batch entry points now exist for basic/adaptive/rigorous on the stronger orthogonal families.
- Tightening: explicit rigorous kernels exist for these families; Chebyshev/Laguerre/Hermite remain weaker and are not in the same engineering state yet.

### Weaker orthogonal families and pfq

- Families: `arb/acb_hypgeom_chebyshev_t`, `arb/acb_hypgeom_chebyshev_u`, `arb/acb_hypgeom_laguerre_l`, `arb/acb_hypgeom_hermite_h`, `arb/acb_hypgeom_pfq`.
- JAX basic: canonical batch paths now have dedicated fixed-shape padded entry points for both real and complex surfaces.
- JAX adaptive/rigorous: staged fixed-shape mode-batch wrappers now exist and now dispatch directly to the family kernels instead of generic mode-wrapper lambdas, but they still rely on generic adaptive/rigorous kernel inflation more than family-specific analytic bounds.
- AD/JAX cleanup: the real Chebyshev kernels now use `lax.cond` instead of `jnp.where` across `arccos/arccosh`, and the complex weaker-orthogonal/pfq kernels now preserve input dtype more consistently.
- Tightening: `pfq` now uses explicit sample tightening on top of the series/tail path; `laguerre_l` and `hermite_h` now use endpoint/corner tightening. These families are improved, but they remain more generic than the stronger orthogonal families and the headline `0f1/1f1/2f1/u` tranche.

### Incomplete gamma

- Families: `arb_hypgeom_gamma_lower`, `arb_hypgeom_gamma_upper`, `acb_hypgeom_gamma_lower`, `acb_hypgeom_gamma_upper`.
- JAX basic: direct sampled evaluation using midpoint-plus-sample boxes.
- Tightening: direct and complementary candidates are now compared explicitly through shared helper builders in `hypgeom.py`, with sampled real/complex parameter boxes instead of midpoint-only evaluation. Adaptive and rigorous mode wrappers now use complement-aware specialized kernels instead of only relying on the generic wrapper inflators.
- Batch/JAX: dedicated fixed-shape padded basic batch entry points now exist for the real and complex lower/upper incomplete gamma families, mode-batch fastpaths are wired in at the API layer, the API now prefers the family-specific padded fastpath for lower/upper incomplete gamma instead of doing generic caller-side padding first, and the family mode-batch cores now dispatch directly to specialized kernels instead of bouncing through generic `*_mode` lambdas.
- AD: boundary sweeps now cover representative real positive-`z` points and complex cut/corner-adjacent points for lower/upper incomplete gamma.
- Compile behavior: family-level compile counts are reduced by the fixed-shape path, and the previous padded `gamma_upper` smoke-benchmark outlier is reduced to ordinary padded overhead. In the broad tranche probe, total compile counts remain `49 -> 47` while family-level counts remain `28 -> 14`; in the targeted incomplete-gamma probe, family compile counts are `4 -> 2` in both `basic` and `rigorous`. Incomplete gamma is improved materially, but still not yet at the same engineering level as base gamma.

### Bessel J/Y/I/K (real/complex)

- Inputs: nu, z (real/complex).
- C Arb: ball arithmetic with series/asymptotics/continued fractions.
- JAX basic: midpoint evaluation + outward rounding (bessel J/Y/I/K use midpoint mode).
- JAX rigorous: derivative bounds + corner sampling (real/complex).
- JAX adaptive: sampling stencil (real/complex).
- SciPy/JAX SciPy: point evaluation (besselj/bessely/besseli/besselk).
- mpmath: arbitrary precision bessel.
- Formulas: series for J/I; Y/K from combinations of J/I.
- Constants: pi, gamma.
- Branch cuts: Y/K along negative real axis; widened if box crosses cut.
- Tightening: corner sampling for `(nu, z)` and explicit handling when `nu` interval crosses integers for Y/K; large‑`z` asymptotic eval added for real J/Y/I/K.

Recent benchmarks:
- Basic mode matches C containment on positive‑`z` tests (5000 samples, warmup timing).
- Rigorous/adaptive containment still low; further analytic bounds needed.
- Code: `src/arbplusjax/hypgeom.py` (bessel section), `src/arbplusjax/ball_wrappers.py` (bessel bounds).

### Higher-level Hankel and spherical Bessel package

- The higher-level complete Hankel and spherical Bessel families now live in `src/arbplusjax/special/bessel/`.
- Hankel:
  - `hankel1`, `hankel2`, `scaled_hankel1`, `scaled_hankel2`
  - direct `J +/- iY` composition plus explicit large-`|z|` asymptotic and derivative/recurrence helpers
- Spherical:
  - `spherical_bessel_j`, `spherical_bessel_y`, `modified_spherical_bessel_i`, `modified_spherical_bessel_k`
  - family-specific series, seed formulas, integer-order recurrences, and large-`|z|` asymptotics
- These are point surfaces today and are exposed through both `arbplusjax.special.bessel` and `arbplusjax.api`.
- Detailed implementation note: `docs/implementation/modules/special_bessel.md`.

## Precision differences and constants

Key differences to track per function:
- Arb uses arbitrary precision for both midpoint and radius.
- JAX uses float64/complex128 midpoint and widens intervals after evaluation.
- JAX rigorous/adaptive bounds depend on derivative sampling or explicit series remainders.
- SciPy/JAX SciPy provide point values only (no enclosure).
- mpmath uses arbitrary precision point evaluation; interval only with `mp.iv`.

Common constants used:
- pi, e, EulerGamma, Bernoulli numbers, zeta values, Lanczos coefficients.
- In JAX, constants are float64 and do not carry arbitrary-precision context.
