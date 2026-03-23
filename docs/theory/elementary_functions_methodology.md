Last updated: 2026-03-16T18:48:00Z

# Elementary Functions Methodology

## Purpose

This note documents the methodology of the helper layer implemented in `src/arbplusjax/elementary.py`. This module is not itself an interval wrapper layer; it provides:

- constants,
- dtype-promotion helpers,
- safe scalar transforms,
- stable logarithmic transforms,
- elementary trigonometric/exponential helper formulas used elsewhere in the library.

The reference software lineage for the surrounding project is `[@Johansson2017]` and `[@flint2026]`, while the JAX runtime substrate is part of the project’s implementation model rather than an external interval backend.

## 1. Role of the elementary layer

The elementary layer does three jobs:

1. provide named mathematical constants in JAX-native dtypes,
2. provide dtype and real/complex promotion rules so higher-level kernels are stable,
3. provide numerically convenient formulas such as

\[
\log(1-e^x), \qquad \log(e^x - 1), \qquad e^{i\pi x}, \qquad z^{-s}.
\]

So this module is methodological infrastructure: it standardizes the point formulas that later appear inside interval, box, and rigorous wrappers.

## 2. Constants and constant-casting helpers

The module defines the following constants:

- `PI`
- `TWO_PI`
- `HALF_PI`
- `LOG_PI`
- `SQRT_PI`
- `LOG_TWO`
- `LOG_TWO_PI`
- `LOG_SQRT_TWO_PI`
- `EULER_GAMMA`
- `TWO_OVER_SQRT_PI`
- `SQRT_TWO_OVER_PI`
- `SQRT_PI_OVER_TWO`
- `E`
- `I`

The methodology is simple: store them once at high fixed precision in JAX scalars, then cast them to the dtype implied by downstream inputs.

The casting helpers are:

- `pi_like`
- `two_pi_like`
- `half_pi_like`
- `log_pi_like`
- `sqrt_pi_like`
- `log_two_like`
- `log_two_pi_like`
- `log_sqrt_two_pi_like`
- `euler_gamma_like`
- `two_over_sqrt_pi_like`
- `sqrt_two_over_pi_like`
- `sqrt_pi_over_two_like`

For each such helper, the methodology is:

\[
c(x_1,\dots,x_k) = \operatorname{cast}(c;\operatorname{dtype}(x_1,\dots,x_k)).
\]

These functions do not change mathematical meaning; they enforce type coherence.

## 3. Real/complex conversion helpers

The conversion and promotion helpers are:

- `as_real`
- `as_complex`
- `promote_dtype`
- `complex_promote`
- `complex_dtype_from`
- `real_dtype_from_complex_dtype`

### `as_real`

Method:

- if input is already real, keep it,
- if input is complex, drop the imaginary part,
- otherwise cast to `float64`.

Conceptually,

\[
\operatorname{as\_real}(x+iy) = x.
\]

### `as_complex` and `complex_promote`

Method:

- if input is real, embed it into the complex line,
- if input is already complex, preserve it.

Conceptually,

\[
\operatorname{as\_complex}(x) = x + 0i.
\]

### `promote_dtype`

Method:

- compute the joint JAX result type of all arguments,
- cast each argument to that type.

This is the point-level prerequisite for stable binary and mixed real/complex formulas.

## 4. Machine-scale helpers

The scale helpers are:

- `eps`
- `tiny`
- `max_value`

These are wrappers around the floating-point metadata of a dtype:

\[
\varepsilon = \text{machine epsilon}, \qquad
\text{tiny} = \text{smallest normal positive number}, \qquad
\text{max} = \text{largest finite number}.
\]

Methodologically, these support guard logic and asymptotic cutoffs in other modules.

## 5. Safe division and log-domain helpers

The numerically sensitive scalar helpers are:

- `safe_div`
- `logaddexp`
- `logsubexp`
- `logsumexp`
- `log1mexp`
- `logexpm1`

### `safe_div`

Method:

\[
\operatorname{safe\_div}(a,b) =
\begin{cases}
a/b, & b \neq 0,\\
\text{fill}, & b = 0.
\end{cases}
\]

This is not a rigorous interval division rule; it is a point-side guard for helper logic.

### `logaddexp`

Method:

\[
\log(e^a + e^b).
\]

Used for stable accumulation in log-space.

### `logsubexp`

Method:

\[
\log(e^a - e^b) = a + \log\!\left(1 - e^{\,b-a}\right),
\]

for the domain where \(a \ge b\).

### `logsumexp`

Method:

\[
\log\sum_i e^{v_i}
= m + \log\sum_i e^{v_i-m},
\qquad m = \max_i v_i.
\]

This is the usual stabilization by factoring out the maximum.

### `log1mexp`

Method:

\[
\log(1-e^x),
\]

implemented with a branch split to avoid catastrophic cancellation near zero.

### `logexpm1`

Method:

\[
\log(e^x - 1),
\]

again with a branch split between small and large \(x\).

## 6. Power and logarithmic magnitude helpers

These helpers are:

- `log_abs`
- `log_pow_abs`
- `x_pow_a`
- `clog`
- `cpow`
- `z_to_minus_s`

### `log_abs`

Method:

\[
\log |z|.
\]

This is a real-valued transform even when \(z\) is complex.

### `log_pow_abs`

Method:

\[
a \log|x|.
\]

This is the real logarithm of \(|x|^a\) when the expression is interpreted on the positive magnitude side.

### `x_pow_a`

Method:

\[
x^a = \exp(a \log x),
\]

with `x` and `a` promoted into the complex domain so branch behavior is controlled by the complex logarithm.

### `clog`

Method:

\[
\operatorname{clog}(z) = \log|z| + i \arg(z),
\]

using the principal argument returned by JAX.

### `cpow`

Method:

\[
z^a = \exp(a\,\operatorname{clog}(z)).
\]

### `z_to_minus_s`

Method:

\[
z^{-s} = \exp(-s\,\operatorname{clog}(z)).
\]

This is especially useful in zeta/polylog/Mellin-Barnes style formulas elsewhere in the project.

## 7. Trigonometric and exponential helper families

These helpers are:

- `cis`
- `sinc`
- `sinc_pi`
- `sin_pi`
- `cos_pi`
- `tan_pi`
- `exp_pi_i`
- `log_sin_pi`

### `cis`

Method:

\[
\operatorname{cis}(x) = \cos x + i \sin x = e^{ix}.
\]

### `sinc`

Method:

\[
\operatorname{sinc}(x) =
\begin{cases}
1, & x \approx 0,\\
\dfrac{\sin x}{x}, & \text{otherwise}.
\end{cases}
\]

The implementation uses a small-threshold branch at the origin.

### `sinc_pi`

Method:

\[
\operatorname{sinc\_pi}(x) =
\begin{cases}
1, & x \approx 0,\\
\dfrac{\sin(\pi x)}{\pi x}, & \text{otherwise}.
\end{cases}
\]

### `sin_pi`, `cos_pi`, `tan_pi`

Methods:

\[
\sin(\pi x), \qquad \cos(\pi x), \qquad \tan(\pi x).
\]

These are explicit helpers because \(\pi x\)-scaled trigonometric arguments occur constantly in reflection, duplication, and special-function identities.

### `exp_pi_i`

Method:

\[
e^{i\pi x}.
\]

### `log_sin_pi`

Method:

\[
\log(\sin(\pi x)).
\]

This is a point-level helper for logarithmic reflection formulas.

## 8. Function inventory

The public elementary functions covered by this methodology are:

- `pi_like`, `two_pi_like`, `half_pi_like`, `log_pi_like`, `sqrt_pi_like`
- `log_two_like`, `log_two_pi_like`, `log_sqrt_two_pi_like`
- `euler_gamma_like`, `two_over_sqrt_pi_like`, `sqrt_two_over_pi_like`, `sqrt_pi_over_two_like`
- `as_real`, `as_complex`, `promote_dtype`, `complex_promote`, `complex_dtype_from`, `real_dtype_from_complex_dtype`
- `eps`, `tiny`, `max_value`
- `safe_div`
- `logaddexp`, `logsubexp`, `logsumexp`, `log1mexp`, `logexpm1`
- `log_abs`, `log_pow_abs`
- `x_pow_a`
- `cis`, `sinc`, `sinc_pi`, `sin_pi`, `cos_pi`, `tan_pi`, `exp_pi_i`, `log_sin_pi`
- `clog`, `cpow`, `z_to_minus_s`

## 9. Local implementation pointers

- [elementary.py](/src/arbplusjax/elementary.py)
- [ball_arithmetic_and_modes.md](/docs/theory/ball_arithmetic_and_modes.md)
- [core_functions_methodology.md](/docs/theory/core_functions_methodology.md)
