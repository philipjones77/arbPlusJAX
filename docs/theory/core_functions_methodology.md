Last updated: 2026-03-16T18:48:00Z

# Core Functions Methodology

## Purpose

This note documents the methodology of the core arithmetic layers:

- real interval core in `src/arbplusjax/arb_core.py`,
- complex box core in `src/arbplusjax/acb_core.py`.

The relevant reference lineage is:

- `[@Johansson2017]` for midpoint-radius interval arithmetic,
- `[@flint2026]` for the FLINT / Arb software family that motivates the API direction.

The key difference, as elsewhere in this project, is that arbPlusJAX implements interval and box semantics over fixed `float64` / `complex128`, not Arb’s arbitrary-precision native ball arithmetic.

## 1. Representation model

### Real core

The real core uses intervals

\[
x = [\ell, u].
\]

Midpoint-radius form is

\[
m = \frac{\ell+u}{2}, \qquad r = \frac{u-\ell}{2}.
\]

### Complex core

The complex core uses boxes

\[
z = [a,b] + i[c,d].
\]

This is a rectangular complex enclosure, not a circular complex ball.

## 2. Methodology inheritance

For both `arb_core` and `acb_core`, the public families follow this pattern:

- base function: point or direct interval/box formula,
- `*_prec`: same computation followed by outward rounding at `prec_bits`,
- `*_batch`: vectorized application of the base function,
- `*_batch_prec`: vectorized application with the same post-hoc widening.

So the mathematical methodology lives at the base-function level. The `prec` and batch variants inherit that methodology.

## 3. Real interval core: `arb_core`

### 3.1 Primary transcendental functions

The primary unary transcendental functions are:

- `arb_exp`
- `arb_log`
- `arb_sqrt`
- `arb_sin`
- `arb_cos`
- `arb_tan`
- `arb_sinh`
- `arb_cosh`
- `arb_tanh`

Methodology:

- `exp`, `sinh`, and `cosh` are monotonic or piecewise monotonic enough that interval endpoints and analytic formulas provide direct interval bounds.
- `log` is monotone on \((0,\infty)\), so

\[
\log([\ell,u]) = [\log \ell, \log u]
\]

on its valid domain.

- `sqrt` is monotone on \([0,\infty)\), so

\[
\sqrt{[\ell,u]} = [\sqrt{\ell}, \sqrt{u}]
\]

on its valid domain.

- `sin`, `cos`, and `tan` require critical-point awareness or branch-safe enclosure logic because monotonicity can fail across the interval.

### 3.2 Basic algebraic operations

The binary and algebraic primitives are:

- `arb_abs`
- `arb_add`
- `arb_sub`
- `arb_mul`
- `arb_div`
- `arb_inv`
- `arb_fma`

Methods:

\[
[a,b] + [c,d] = [a+c, b+d],
\]

\[
[a,b] - [c,d] = [a-d, b-c],
\]

\[
[a,b][c,d] =
[\min(ac,ad,bc,bd), \max(ac,ad,bc,bd)],
\]

with outward rounding.

Division is implemented through reciprocal logic when zero is excluded from the denominator interval:

\[
\frac{[a,b]}{[c,d]} = [a,b]\cdot \frac{1}{[c,d]}.
\]

The fused multiply-add methodology is:

\[
\operatorname{fma}(x,y,z) = xy + z
\]

with interval multiplication and interval addition composed in one kernel.

### 3.3 Cancellation-aware elementary transforms

These are:

- `arb_log1p`
- `arb_expm1`
- `arb_sin_cos`
- `arb_sinh_cosh`

Methods:

\[
\log1p(x) = \log(1+x), \qquad \expm1(x) = e^x - 1,
\]

with formulas chosen to improve numerical behavior near the origin.

`arb_sin_cos` and `arb_sinh_cosh` compute paired outputs so shared work and common substructure can be preserved.

### 3.4 Pi-scaled trigonometric family

These are:

- `arb_sin_pi`
- `arb_cos_pi`
- `arb_tan_pi`
- `arb_sinc`
- `arb_sinc_pi`

Methods:

\[
\sin(\pi x), \qquad \cos(\pi x), \qquad \tan(\pi x),
\]

\[
\operatorname{sinc}(x)=\frac{\sin x}{x}, \qquad
\operatorname{sinc\_pi}(x)=\frac{\sin(\pi x)}{\pi x},
\]

with regularized origin handling for the sinc variants.

### 3.5 Inverse trigonometric and inverse hyperbolic family

These are:

- `arb_asin`
- `arb_acos`
- `arb_atan`
- `arb_asinh`
- `arb_acosh`
- `arb_atanh`

Methods:

\[
\arcsin(x),\ \arccos(x),\ \arctan(x),\ \operatorname{asinh}(x),\ \operatorname{acosh}(x),\ \operatorname{atanh}(x)
\]

using interval propagation through the corresponding JAX point formulas together with outward rounding.

The key methodology issue here is domain management:

- `acosh` requires \(x \ge 1\),
- `atanh` excludes intervals crossing \(\pm 1\) without appropriate widening.

### 3.6 Sign, powers, and roots

These are:

- `arb_sign`
- `arb_pow`
- `arb_pow_ui`
- `arb_pow_fmpz`
- `arb_pow_fmpq`
- `arb_root_ui`
- `arb_root`
- `arb_cbrt`

Methods:

\[
\operatorname{sign}(x),
\qquad
x^y,
\qquad
x^n,
\qquad
x^{p/q},
\qquad
\sqrt[k]{x}.
\]

For integer exponents the methodology is simpler because repeated multiplication or parity structure can be exploited. For rational or general exponents, branch and domain constraints matter more strongly.

### 3.7 Log-gamma family

These are:

- `arb_lgamma`
- `arb_gamma`
- `arb_rgamma`

Methods:

\[
\log \Gamma(x), \qquad \Gamma(x), \qquad \frac{1}{\Gamma(x)}.
\]

These are core in the API surface, but methodologically they sit near the boundary between elementary and special functions. In the implementation, they use explicit scalar formulas and shared approximations, then inherit the same `prec` and batch widening rules as the other core kernels.

### 3.8 Variant families

For every base real-core function above, there are corresponding:

- `*_prec`
- `*_batch`
- `*_batch_prec`

families. Methodologically:

- `*_prec` means midpoint/direct interval computation plus post-hoc outward widening,
- `*_batch` means the same formula under vectorization,
- `*_batch_prec` means vectorization plus post-hoc widening.

## 4. Complex box core: `acb_core`

### 4.1 Structural and predicate helpers

These are:

- `as_acb_box`
- `acb_box`
- `acb_real`
- `acb_imag`
- `acb_midpoint`
- `acb_zero`
- `acb_one`
- `acb_onei`
- `acb_is_exact`
- `acb_is_zero`
- `acb_is_one`
- `acb_is_real`
- `acb_is_int`
- `acb_set`

Methodology:

- construct, inspect, or classify rectangular complex boxes,
- extract midpoint or real/imaginary interval components,
- provide identity elements and predicates needed by higher-level kernels.

### 4.2 Basic algebra on complex boxes

These are:

- `acb_neg`
- `acb_conj`
- `acb_add`
- `acb_sub`
- `acb_mul`
- `acb_mul_naive`
- `acb_mul_onei`
- `acb_div_onei`
- `acb_inv`
- `acb_div`

Methods:

For

\[
z = a+ib, \qquad w = c+id,
\]

the multiplication formula is

\[
zw = (ac-bd) + i(ad+bc),
\]

and division is

\[
\frac{z}{w} = \frac{z\bar{w}}{|w|^2}.
\]

All of these are propagated through interval arithmetic on the real and imaginary components.

### 4.3 Magnitude, argument, and real-projection helpers

These are:

- `acb_abs`
- `acb_arg`
- `acb_sgn`
- `acb_csgn`
- `acb_real_abs`
- `acb_real_sgn`
- `acb_real_heaviside`
- `acb_real_floor`
- `acb_real_ceil`
- `acb_real_max`
- `acb_real_min`
- `acb_real_sqrtpos`

Methodology:

- derive real interval bounds from boxes,
- expose real-valued projections used inside special-function logic and branch control.

### 4.4 Analytic helper kernels

These are:

- `acb_sqrt_analytic`
- `acb_rsqrt_analytic`
- `acb_log_analytic`
- `acb_pow_analytic`

These encode direct analytic formulas on boxes for the corresponding complex functions, and they are central to the project’s “rigorous where available” methodology.

### 4.5 Dot-product and accumulation family

These are:

- `acb_dot_simple`
- `acb_dot_precise`
- `acb_dot`
- `acb_approx_dot`
- `acb_dot_ui`
- `acb_dot_si`
- `acb_dot_uiui`
- `acb_dot_siui`
- `acb_dot_fmpz`

Methodology:

\[
\sum_k z_k w_k
\]

with different contracts for coefficient type and accumulation policy.

### 4.6 Error and enclosure-management helpers

These are:

- `acb_add_error_arb`
- `acb_add_error_arf`
- `acb_add_error_mag`
- `acb_union`
- `acb_trim`

Methodology:

- enlarge or combine boxes,
- explicitly attach additional real error bounds,
- trim redundant enclosure slack when possible.

### 4.7 Primary transcendental family

These are:

- `acb_exp`
- `acb_log`
- `acb_log1p`
- `acb_sqrt`
- `acb_rsqrt`
- `acb_sin`
- `acb_cos`
- `acb_sin_cos`
- `acb_tan`
- `acb_cot`
- `acb_sinh`
- `acb_cosh`
- `acb_tanh`
- `acb_asin`
- `acb_acos`
- `acb_atan`
- `acb_asinh`
- `acb_acosh`
- `acb_atanh`
- `acb_sech`
- `acb_csch`

Methods:

- direct complex point formulas evaluated at the box midpoint or through analytic box formulas,
- then outward enclosure recovery at the box level,
- with special branch handling for `log`, inverse functions, and reciprocal functions.

### 4.8 Pi-scaled and normalized trigonometric family

These are:

- `acb_sin_pi`
- `acb_cos_pi`
- `acb_sin_cos_pi`
- `acb_tan_pi`
- `acb_cot_pi`
- `acb_csc_pi`
- `acb_sinc`
- `acb_sinc_pi`
- `acb_exp_pi_i`
- `acb_expm1`
- `acb_exp_invexp`

Methods:

\[
\sin(\pi z),\ \cos(\pi z),\ \tan(\pi z),\ e^{i\pi z},
\]

plus normalized or paired-output variants that improve stability and code reuse.

### 4.9 Multiplicative transform family

These are:

- `acb_addmul`
- `acb_submul`
- `acb_pow`
- `acb_pow_arb`
- `acb_pow_ui`
- `acb_pow_si`
- `acb_pow_fmpz`
- `acb_sqr`
- `acb_root_ui`

Methods:

\[
xy+z, \qquad xy-z, \qquad z^w, \qquad z^n, \qquad z^2, \qquad \sqrt[k]{z}.
\]

These are the core algebraic building blocks used by more specialized families.

### 4.10 Gamma/zeta/polylog family inside the core layer

These are:

- `acb_gamma`
- `acb_rgamma`
- `acb_lgamma`
- `acb_log_sin_pi`
- `acb_digamma`
- `acb_zeta`
- `acb_hurwitz_zeta`
- `acb_polygamma`
- `acb_bernoulli_poly_ui`
- `acb_polylog`
- `acb_polylog_si`

Methodology:

These functions are mathematically beyond “elementary”, but they sit in `acb_core.py` because the complex box layer is the natural place to host generic complex formulas. They inherit the same box-first methodology:

- midpoint or analytic complex evaluation,
- box widening to preserve containment,
- branch-aware handling where singularities or cuts are present.

### 4.11 AGM family

These are:

- `acb_agm`
- `acb_agm1`
- `acb_agm1_cpx`

Methods:

Arithmetic-geometric mean iterations, with complex-aware variants for the principal branch setting.

### 4.12 Precision variants

The complex precision and widening family includes:

- `acb_box_round_prec`
- `acb_exp_prec`, `acb_log_prec`, `acb_sqrt_prec`
- `acb_sin_prec`, `acb_cos_prec`, `acb_tan_prec`
- `acb_sinh_prec`, `acb_cosh_prec`, `acb_tanh_prec`
- `acb_asin_prec`, `acb_acos_prec`, `acb_atan_prec`
- and the corresponding continuation further down the module for the other base functions

Methodologically, these apply the same base-function formula and then widen the output box according to `prec_bits`.

## 5. Mode interaction

The core modules themselves expose the base interval/box formulas. The four-mode interpretation is layered above them:

- `point`: point wrappers only, no interval/box semantics,
- `basic`: midpoint or direct interval/box core formula plus outward rounding,
- `adaptive`: wrapper-level sampling tightening,
- `rigorous`: analytic interval/box formulas where available, otherwise derivative/Jacobian-style enclosure at the wrapper layer.

So the core modules provide the arithmetic substrate; the wrappers provide the mode semantics.

## 6. Function inventory summary

### Real core base functions

- `arb_exp`, `arb_log`, `arb_sqrt`
- `arb_sin`, `arb_cos`, `arb_tan`
- `arb_sinh`, `arb_cosh`, `arb_tanh`
- `arb_abs`, `arb_add`, `arb_sub`, `arb_mul`, `arb_div`, `arb_inv`, `arb_fma`
- `arb_log1p`, `arb_expm1`, `arb_sin_cos`, `arb_sinh_cosh`
- `arb_sin_pi`, `arb_cos_pi`, `arb_tan_pi`, `arb_sinc`, `arb_sinc_pi`
- `arb_asin`, `arb_acos`, `arb_atan`, `arb_asinh`, `arb_acosh`, `arb_atanh`
- `arb_sign`
- `arb_pow`, `arb_pow_ui`, `arb_pow_fmpz`, `arb_pow_fmpq`
- `arb_root_ui`, `arb_root`, `arb_cbrt`
- `arb_lgamma`, `arb_gamma`, `arb_rgamma`

Each of these has corresponding `*_prec`, `*_batch`, and `*_batch_prec` families.

### Complex core base functions

- structural/predicate layer: `as_acb_box`, `acb_box`, `acb_real`, `acb_imag`, `acb_midpoint`, `acb_zero`, `acb_one`, `acb_onei`, `acb_is_exact`, `acb_is_zero`, `acb_is_one`, `acb_is_real`, `acb_is_int`, `acb_set`
- algebra layer: `acb_neg`, `acb_conj`, `acb_add`, `acb_sub`, `acb_mul`, `acb_mul_naive`, `acb_mul_onei`, `acb_div_onei`, `acb_inv`, `acb_div`
- real-projection layer: `acb_abs`, `acb_arg`, `acb_sgn`, `acb_csgn`, `acb_real_abs`, `acb_real_sgn`, `acb_real_heaviside`, `acb_real_floor`, `acb_real_ceil`, `acb_real_max`, `acb_real_min`, `acb_real_sqrtpos`
- analytic helpers: `acb_sqrt_analytic`, `acb_rsqrt_analytic`, `acb_log_analytic`, `acb_pow_analytic`
- dot/error/enclosure layer: `acb_dot_simple`, `acb_dot_precise`, `acb_dot`, `acb_approx_dot`, `acb_dot_ui`, `acb_dot_si`, `acb_dot_uiui`, `acb_dot_siui`, `acb_dot_fmpz`, `acb_add_error_arb`, `acb_add_error_arf`, `acb_add_error_mag`, `acb_union`, `acb_trim`
- transcendental layer: `acb_exp`, `acb_log`, `acb_log1p`, `acb_sqrt`, `acb_rsqrt`, `acb_sin`, `acb_cos`, `acb_sin_cos`, `acb_tan`, `acb_cot`, `acb_sinh`, `acb_cosh`, `acb_tanh`, `acb_asin`, `acb_acos`, `acb_atan`, `acb_asinh`, `acb_acosh`, `acb_atanh`, `acb_sech`, `acb_csch`
- pi-scaled layer: `acb_sin_pi`, `acb_cos_pi`, `acb_sin_cos_pi`, `acb_tan_pi`, `acb_cot_pi`, `acb_csc_pi`, `acb_sinc`, `acb_sinc_pi`, `acb_exp_pi_i`, `acb_expm1`, `acb_exp_invexp`
- multiplicative transforms: `acb_addmul`, `acb_submul`, `acb_pow`, `acb_pow_arb`, `acb_pow_ui`, `acb_pow_si`, `acb_pow_fmpz`, `acb_sqr`, `acb_root_ui`
- gamma/zeta/polylog layer: `acb_gamma`, `acb_rgamma`, `acb_lgamma`, `acb_log_sin_pi`, `acb_digamma`, `acb_zeta`, `acb_hurwitz_zeta`, `acb_polygamma`, `acb_bernoulli_poly_ui`, `acb_polylog`, `acb_polylog_si`
- AGM layer: `acb_agm`, `acb_agm1`, `acb_agm1_cpx`

These then continue into the module’s precision and batched variants.

## 7. Local implementation pointers

- [arb_core.py](/src/arbplusjax/arb_core.py)
- [acb_core.py](/src/arbplusjax/acb_core.py)
- [arb_core.md](/docs/implementation/modules/arb_core.md)
- [acb_core.md](/docs/implementation/modules/acb_core.md)
- [elementary_functions_methodology.md](/docs/theory/elementary_functions_methodology.md)
