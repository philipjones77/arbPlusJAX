Last updated: 2026-03-16T18:10:00Z

# Ball Arithmetic and the Four-Mode Model

## Purpose

This note defines the mathematical model behind classical ball arithmetic and then compares it to the four execution modes used throughout arbPlusJAX:

- `point`
- `basic`
- `adaptive`
- `rigorous`

The key distinction is that Arb / FLINT performs **arbitrary-precision ball arithmetic during the computation**, while arbPlusJAX uses **float64 or complex128 midpoint evaluation plus interval widening on top of that substrate**.

## 1. Ball arithmetic in Arb / FLINT

In ball arithmetic, a real quantity is represented as

\[
x = m \pm r,
\]

meaning the closed interval

\[
[m-r,\; m+r].
\]

For a complex value, Arb uses a rectangular complex enclosure whose real and imaginary parts are each real balls. In FLINT documentation, `arb_t` is the real ball type and `acb_t` is the complex type built from real balls.

The semantic guarantee is enclosure: if the input balls contain the exact mathematical inputs, then the output ball contains the exact mathematical result. The output enclosure is generally not minimal, but it is designed to be rigorous.

Two implementation facts from Arb matter for comparison with this repository:

- The **midpoint** is stored in an arbitrary-precision floating-point type (`arf_t`).
- The **radius / error bound** is stored separately as a magnitude bound (`mag_t`).

So Arb is not merely “interval arithmetic with a midpoint notation”; it is a full arithmetic system in which approximation error, truncation error, and rounding error are propagated together during every operation.

## 2. Why ball arithmetic is attractive

Ball arithmetic is useful because it separates:

- a central approximation \(m\), which can be evaluated efficiently, and
- a nonnegative error radius \(r\), which certifies the uncertainty.

For a function \(f\), the ideal enclosure statement is

\[
f(x) \subseteq f(m \pm r) \subseteq y \pm s,
\]

where \(y \pm s\) is the computed output ball. In practical implementations, \(s\) includes:

- propagated input uncertainty,
- floating-point rounding error,
- truncation error from series or asymptotic expansions,
- algorithmic remainder bounds.

This is the model followed by Arb’s special-function algorithms, including its rigorous hypergeometric machinery.

## 3. arbPlusJAX uses boxes, not native Arb balls

arbPlusJAX does **not** reimplement Arb’s internal number system. Its base enclosure objects are:

- real intervals as `[lo, hi]`,
- complex boxes as `[re_lo, re_hi, im_lo, im_hi]`.

The numeric substrate is standard JAX floating point:

- `float64` for real midpoint work,
- `complex128` for complex midpoint work.

This means `prec_bits` in arbPlusJAX does **not** switch the computation to a true arbitrary-precision arithmetic. Instead, `prec_bits` controls how aggressively the final enclosure is widened by outward rounding or mode-specific inflation.

That difference is the central theoretical gap between Arb and this project.

## 4. The arbPlusJAX four-mode model

The repository standardizes four modes across wrappers and public APIs.

### 4.1 `point`

`point` mode ignores interval semantics and computes only a point value:

\[
y = f(x).
\]

This is the throughput-oriented baseline. It is useful for:

- pure numerical evaluation,
- batched/JIT compilation with the smallest state,
- comparisons against interval-enabled modes.

It offers no enclosure guarantee.

### 4.2 `basic`

`basic` mode starts from an interval \(x = [\ell, u]\), forms the midpoint

\[
m = \frac{\ell + u}{2},
\]

evaluates the function at that midpoint,

\[
y = f(m),
\]

and then converts the point result back to an interval or box by outward rounding.

Conceptually, this is:

\[
[\operatorname{below}(y),\; \operatorname{above}(y)].
\]

This is fast and JIT-friendly, but it is only a weak analogue of ball arithmetic because:

- the function is evaluated only at the midpoint,
- nonlinearity across the whole input interval is not tracked,
- internal arithmetic precision is still ordinary `float64` / `complex128`.

So `basic` should be read as “interval-shaped output around a point computation”, not as a full rigorous interval method.

### 4.3 `adaptive`

`adaptive` mode also begins with a midpoint-radius view of the input, but uses a fixed stencil of sample points around the midpoint:

\[
z_i = m + r t_i.
\]

The output radius is estimated from sampled deviations,

\[
r_{\mathrm{out}} \approx \max_i |f(z_i) - f(m)| + \varepsilon,
\]

or the analogous box/Jacobian construction in complex settings.

The purpose of `adaptive` mode is pragmatic:

- tighten enclosures relative to `basic`,
- keep the sample shape static for JAX compilation,
- avoid dynamic control flow that would fragment compilation caches.

This mode is often useful in practice, but it is still heuristic unless the sampling argument can be promoted to a proven bound for the specific kernel.

### 4.4 `rigorous`

`rigorous` mode is the closest arbPlusJAX analogue to ball arithmetic. It interprets the input as a midpoint plus radius and then tries to propagate a certified or conservative output radius.

When analytic bounds are available, the implementation uses direct formulas or monotonicity arguments. Otherwise, it falls back to derivative, Jacobian, or Lipschitz-style bounds around the midpoint. In simplified form:

\[
|f(x) - f(m)| \le L r,
\]

so the output radius is taken as

\[
r_{\mathrm{out}} = L r + \varepsilon.
\]

For complex functions, the same idea is applied through Jacobian norms or equivalent first-order bounds.

This is substantially closer to the Arb philosophy than `basic` or `adaptive`, but it is still not identical to Arb because:

- the midpoint computation is still fixed-precision JAX arithmetic,
- some bounds are wrapper-level rather than intrinsic to every primitive operation,
- some families use sampled derivative bounds instead of Arb-style end-to-end arbitrary-precision remainder tracking.

Therefore, `rigorous` in arbPlusJAX means “the strongest containment-oriented mode provided by this project”, not “an exact clone of Arb semantics”.

## 5. Comparison with classical ball arithmetic

The cleanest way to compare the systems is by asking where the enclosure is enforced.

### Arb / FLINT

Enclosure is enforced **inside the arithmetic itself**:

- midpoint arithmetic uses arbitrary precision,
- the radius is updated during every primitive operation,
- algorithm-specific truncation and remainder bounds are integrated directly into the returned ball.

### arbPlusJAX

Enclosure is enforced **around float64 / complex128 kernels**:

- `point` has no enclosure,
- `basic` wraps a midpoint value with outward rounding,
- `adaptive` widens using fixed sampling,
- `rigorous` uses the best available analytic or derivative-style bound on top of point kernels.

So arbPlusJAX is best understood as a JAX-native interval framework inspired by Arb, not as a replacement for Arb’s arbitrary-precision ball arithmetic.

## 6. Practical interpretation for this project

For this repository, the four modes serve different engineering goals:

- `point`: fastest pure numerical path.
- `basic`: cheapest interval-shaped API compatible with the rest of the stack.
- `adaptive`: tighter practical enclosure when cheap midpoint rounding is too loose.
- `rigorous`: preferred containment-oriented mode when a proof-style interval result is needed from the JAX implementation.

The right mental model is:

- Arb answers “how do we do rigorous arithmetic natively?”
- arbPlusJAX answers “how do we expose point and interval semantics in a JAX-first system while preserving compilation stability and useful error control?”

## 7. References

References available in the local bibliography library:

- `[@Johansson2017]` for Arb’s midpoint-radius interval arithmetic model.
- `[@flint2026]` for the FLINT / Arb software and documentation lineage used as the external scalar reference in this project.
- `[@mpmath2023]` for the arbitrary-precision point-arithmetic comparison backend.

Bibliography source used for these keys:

- `/mnt/c/dev/references/bibliography/library/software.bib`

The current bibliography snapshot does not appear to contain a separate entry for Johansson’s hypergeometric-rigorous paper, so that one is not yet cited by key here.

## 8. Local implementation pointers

For the concrete arbPlusJAX definitions referenced in this note, see:

- [precision_standard.md](/docs/standards/precision_standard.md)
- [governance/architecture.md](/docs/governance/architecture.md)
- [ball_wrappers.md](/docs/implementation/wrappers/ball_wrappers.md)
- [baseline_wrappers.md](/docs/implementation/wrappers/baseline_wrappers.md)
- [hypgeom_wrappers.md](/docs/implementation/wrappers/hypgeom_wrappers.md)
