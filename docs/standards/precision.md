Last updated: 2026-02-25T03:51:38Z

# Precision and Interval Semantics

## Summary

arbPlusJAX is JAX-first and uses **float64 interval boxes** with outward rounding. Precision control is modeled by widening bounds after computation. This differs from Arb’s **ball arithmetic**, which tracks midpoint and radius with arbitrary precision through every operation.

We standardize **four modes** across modules:

- **point**: pure point evaluation (no intervals, no bounds)
- **basic**: midpoint evaluation + outward rounding (fastest interval mode)
- **adaptive**: fixed-grid sampling around the midpoint with extra inflation (non‑recompilable heuristic)
- **rigorous**: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint

## Precision comparison (mpmath vs JAX vs Arb vs jax.special)

| System | Precision model | How to set | Notes |
|---|---|---|---|
| mpmath | Arbitrary precision (decimal/bits) | `mp.mp.dps`, `mp.mp.prec` | Real/complex high precision; interval mode via `mp.iv`. |
| arbPlusJAX | Fixed float64/complex128 | `prec_bits` (outward rounding only) | No true arbitrary precision; `prec_bits` widens bounds post‑hoc. |
| Arb (C) | Arbitrary precision ball arithmetic | `prec` (bits) per call | Rigorous midpoint + radius propagation. |
| jax.special | Fixed float64/complex128 | dtype (float32/64) | Same numeric precision as JAX; no interval/ball semantics. |
| Mathematica | Arbitrary precision (decimal/bits) | `WorkingPrecision` | High‑precision point evaluation; no interval enclosures in our harness. |

## Core methodology differences (math and semantics)

### C Arb (ball arithmetic)

Arb represents values as **balls**:
\[
 x = m \pm r
\]
- \(m\) and \(r\) are arbitrary-precision floating point values.
- Every operation updates both \(m\) and \(r\) with rigorously tracked rounding/truncation error.
- Precision \(p\) (bits) is passed per-call and used throughout the full computation.

### arbPlusJAX basic (point + outward rounding)

Represent intervals as \([l, u]\) and collapse to the midpoint \(m\):
\[
 m = (l+u)/2, \quad y = f(m)
\]
- Convert to an output interval via outward rounding.
- Optional `prec_bits` **inflates the interval** after the fact:
\[
 [\text{below}(y), \text{above}(y)] \rightarrow \text{round\_outward}(\cdot, \text{prec\_bits})
\]
- No internal arbitrary precision; only float64/complex128.

### arbPlusJAX rigorous (analytic or Jacobian/Lipschitz bounds)

When a **closed-form bound** is available (exp/log/sin/cos/tan/sinh/cosh/tanh/sqrt):
- Use monotonicity or analytic formulas to bound outputs directly.

Otherwise:
\[
 \|f(x) - f(m)\| \le \|J_f(m)\| \cdot r
\]
- Use Jacobian/Lipschitz bounds around midpoint kernels.
- Enclosure uses outward rounding + safety inflation.

### arbPlusJAX adaptive (sampling)

Use a fixed stencil around the midpoint:
\[
 z_i = m + r t_i
\]
- Set radius by sampled deviations:
\[
 \mathrm{rad}_{\text{out}} = \max_i |f(z_i) - f(m)| + \varepsilon
\]
- Tight in practice, not guaranteed rigorous for all functions/domains.

### SciPy / JAX SciPy / JAX NumPy

Point evaluation only:
\[
 y = f(x)
\]
- Floating‑point point values only.
- No interval or ball semantics.

### mpmath

High precision point evaluation:
\[
 y = f(x) \quad \text{with } p \text{ decimal digits}
\]
- Arbitrary precision with `mp.mp.dps`/`mp.mp.prec`.
- No interval enclosures unless using `mp.iv`.

## mpmath-style interface in arbPlusJAX

For JAX compatibility we provide a **mpmath-like API** that maps `dps`/`prec` to `prec_bits` and only **inflates bounds** after float64 midpoint evaluation:

- `precision.set_dps(dps)` / `precision.set_prec_bits(prec_bits)`
- `precision.workdps(dps)` / `precision.workprec(prec_bits)`
- `ball_wrappers.*_mp(...)` helpers for `exp`, `log`, `sin`, `gamma`

This does **not** change the underlying arithmetic precision; it only changes the outward rounding inflation.

## Methodologies with math

### 1) basic JAX interval kernels (arbPlusJAX)

Let the interval be \(x=[\ell,u]\), midpoint \(m=(\ell+u)/2\).

Compute:
\[
 y = f(m)
\]
Then return a box using outward rounding:
\[
 \mathrm{out} = [\mathrm{below}(y),\ \mathrm{above}(y)]
\]
Precision control (if `prec_bits` is provided) applies **post‑hoc widening**:
\[
 \mathrm{out}_{\text{prec}} = \mathrm{round\_outward}(\mathrm{out},\ \text{prec\_bits})
\]

This is **not** rigorous unless the function is locally linear and the rounding inflation is sufficient.

### 2) Rigorous ball wrapper (arbPlusJAX)

Represent the interval as a **ball**: midpoint \(m\), radius \(r=(u-\ell)/2\).

Estimate a Lipschitz bound using sampled derivatives:
\[
 L \approx \max_i |f'(m + r t_i)|
\]
Then:
\[
 \mathrm{rad}_{\text{out}} = L r + \varepsilon,\quad \varepsilon = 2^{-\text{prec\_bits}}
\]
Return:
\[
 [f(m)-\mathrm{rad}_{\text{out}},\ f(m)+\mathrm{rad}_{\text{out}}]
\]

Complex uses the Jacobian norm:
\[
 L \approx \max_i \|J_f(m + r e^{i\theta_i})\|_F
\]

### 3) Adaptive ball wrapper (arbPlusJAX, non‑recompilable)

Sample a fixed grid (static length, no recompilation):
\[
 z_i = m + r \cdot t_i
\]
and set:
\[
 \mathrm{rad}_{\text{out}} = \max_i |f(z_i) - f(m)| + \varepsilon
\]

Same return box as above. This is a heuristic but often tighter than basic for nonlinear functions.

### 4) Arb (C) ball arithmetic

Arb represents values as:
\[
 x = m \pm r
\]
and propagates \(r\) rigorously through each operation using arbitrary precision. The precision \(p\) (bits) is specified per call; internal rounding and truncation errors are folded into \(r\).

### 5) mpmath

mpmath computes:
\[
 y = f(m)
\]
at arbitrary precision `mp.dps` or `mp.prec`, but **does not** produce enclosures unless you use the interval context `mp.iv`.

### 6) JAX point intrinsics

Public JAX point intrinsics such as `jax.lax` and `jax.numpy` are standard floating-point evaluations with dtype-level precision (float32/float64). No enclosure, no error bounds.

## Data types

- **Real interval**: `[lo, hi]` as `float64` shape `(2,)`.
- **Complex box**: `[re_lo, re_hi, im_lo, im_hi]` as `float64` shape `(4,)`.
- **Midpoint eval**: `float64` (real) or `complex128` (complex).
- **Integer parameters**: `int64`.

## Precision rules

Two modes are used consistently:

1. **Default (non‑prec)**  
   - Evaluate the function at the midpoint.
   - Convert to interval by outward rounding using `_below/_above`.

2. **Precision-aware (`*_prec`)**  
   - Same midpoint evaluation.
   - Apply `double_interval.round_interval_outward` with `prec_bits`.
   - Lower `prec_bits` yields a **wider** interval; higher `prec_bits` yields a **tighter** interval.

### Batch/JIT

- `*_batch`, `*_batch_prec`, and `*_batch_prec_jit` apply the same semantics after vectorization.

## Error handling

- If any computation returns non‑finite values, the result is set to a full interval or full complex box.
- Functions with restricted domains map invalid inputs to full intervals/boxes.

## How this differs from Arb

### Arb (C)

- Uses **ball arithmetic**: value = midpoint ± radius.
- Midpoint and radius are tracked at **arbitrary precision**.
- Error bounds are propagated rigorously through each operation.

### arbPlusJAX

- Uses **interval boxes**.
- Evaluates at the midpoint in **float64** (or complex128), then expands bounds.
- Precision is modeled **after the fact**, not during each operation.

## Practical implications

- Arb’s bounds are typically tighter and more reliable for difficult domains.
- arbPlusJAX is conservative but not as rigorous for series truncations or near singularities.
- Use parity tests and benchmarks to quantify differences for each module.

## Recent empirical findings (2026-02-25)

- **loggamma**: JAX basic (interval midpoint) matches C‑Arb midpoint to ~1e‑15 on real/complex ranges tested; JAX point and mpmath match each other but differ from C midpoint by ~1e‑3 in the wider range.  
- **bessel J/Y/I/K**: basic mode now uses midpoint evaluation and matches C containment on positive‑z tests; rigorous/adaptive still have low containment and need tighter analytic bounds.

## Four-mode semantics (current standard)

### point

Point-only kernels return float64/complex128 values without interval semantics:
\[
 y = f(x)
\]
This is the fastest mode and the baseline for performance comparisons.

### basic

For any `*_prec` kernel we compute midpoint output and apply outward rounding:
\[
 y = f(m), \quad \mathrm{out} = \mathrm{round\_outward}(y, \text{prec\_bits})
\]

### Rigorous (analytic where available)

For core functions with known identities (exp/log/sin/cos/tan/sinh/cosh/tanh, sqrt):

- Real: monotonicity + critical point checks.
- Complex: analytic formulas (e.g. \(\exp(x+iy)=e^x(\cos y+i\sin y)\)) with interval propagation.
- Branch cuts handled by widening (e.g. `log` imag part widened to \([‑\pi,\pi]\) if the box crosses the negative real axis or contains 0).
- Sqrt uses polar form: \(\sqrt{z}=\sqrt{r}(\cos(\theta/2)+i\sin(\theta/2))\) with interval \(\theta\).

### Rigorous (Jacobian/Lipschitz bounds for general kernels)

For modules where analytic bounds are not practical (`dirichlet`, `calc`, `mat`, `poly`, `modular`, `elliptic`), we use a Jacobian-based bound on midpoint kernels:

Let \(g\) be the midpoint kernel (intervals collapsed to midpoints). For input vector \(x\):
\[
 g(x) \in \mathbb{R}^m,\quad J = \frac{\partial g}{\partial x}
\]
Given midpoint \(m\) and componentwise radii \(r\), we bound:
\[
 |g(x) - g(m)| \le |J(m)| \, r
\]
and return outward-rounded intervals for each output component.

For complex boxes, we treat real and imaginary parts as separate variables and bound both components using the same Jacobian framework.

For `calc`, keep the following distinction explicit:

- calc names such as `integrate_line`, `gl_auto_deg`, and `taylor` refer to numerical method families
- `point`, `basic`, `adaptive`, and `rigorous` are arbPlusJAX dispatch/tightening modes applied by `calc_wrappers` and `api`

So `gl_auto_deg` and `taylor` are not additional modes; they are alternative calc kernels that can be exposed through the four-mode interface.

### Adaptive (sampling)

We evaluate the midpoint kernel at a small fixed stencil:
\[
 \{m,\ m+r,\ m-r\}
\]
and set:
\[
 \mathrm{rad}_{\text{out}} = \max_i |g(x_i)-g(m)| + 2^{-\text{prec\_bits}}
\]

This is not fully rigorous but usually tighter than basic in nonlinear regions. It is static‑shape and JIT‑friendly.

### Where this is implemented

- **Core (analytic bounds)**: `arb_core`, `acb_core` via `core_wrappers`.
- **Generic bound tightening**: `dirichlet_wrappers`, `calc_wrappers`, `mat_wrappers`, `poly_wrappers`, `modular_elliptic_wrappers` use Jacobian/sampling bounds.
- **DFT/Convolution (analytic interval arithmetic)**: `dft_wrappers` uses explicit interval twiddle factors and interval linear algebra in rigorous mode.
- **Ball wrappers**: `ball_wrappers` for exp/log/sin/gamma with derivative/sampling bounds.

## Tightening methodology (function-specific)

Tightening refers to **domain-aware bounding** beyond generic Jacobian or sampling bounds. Examples:

- **Hypergeometric U (`acb_hypgeom_u`)**: corner/endpoint sampling and regime switching for asymptotic vs. series behavior.  
  Code: `src/arbplusjax/hypgeom.py` (search `_real_hypu_regime`, `_complex_hypu_regime`, `acb_hypgeom_u`).

- **0F1/1F1/2F1**: explicit series tail remainder bounds with endpoint sampling.  
  Code: `src/arbplusjax/hypgeom.py` (search `*_tail_bound`, `acb_hypgeom_0f1`, `acb_hypgeom_1f1`, `acb_hypgeom_2f1`).

- **Legendre/Jacobi/Gegenbauer helpers**: endpoint/corner sampling for stability near singular points.  
  Code: `src/arbplusjax/hypgeom.py` (search `legendre`, `jacobi`, `gegenbauer`).

- **Gamma lower/upper regularized forms**: complementary relationship used for numerical stability.  
  Code: `src/arbplusjax/hypgeom.py` (search `gamma_lower`, `gamma_upper`).

## Code-level references by methodology

Use these references to document per-function behavior in audits or module notes.

### C Arb (ball arithmetic)
- C reference libs: `stuff/migration/c_chassis` (archived build scripts and DLLs).
- The actual C implementation lives in the flint repo; arbPlusJAX links via `tests/_arb_c_chassis.py`.

### basic JAX (midpoint + outward rounding)
- Interval kernels: `src/arbplusjax/arb_core.py`, `src/arbplusjax/acb_core.py`
- Post‑hoc precision widening: `src/arbplusjax/double_interval.py` (`round_interval_outward`)
- basic dispatch (mp-mode): `src/arbplusjax/baseline_wrappers.py`

### Rigorous JAX (analytic / Jacobian / Lipschitz)
- Analytic bounds for core: `src/arbplusjax/core_wrappers.py`
- Jacobian/Lipschitz wrappers:  
  `src/arbplusjax/dirichlet_wrappers.py`, `src/arbplusjax/calc_wrappers.py`,  
  `src/arbplusjax/mat_wrappers.py`, `src/arbplusjax/poly_wrappers.py`,  
  `src/arbplusjax/modular_elliptic_wrappers.py`, `src/arbplusjax/dft_wrappers.py`

### Adaptive JAX (sampling)
- Ball sampling wrappers: `src/arbplusjax/ball_wrappers.py`
- Adaptive dispatch in mp-mode: `src/arbplusjax/baseline_wrappers.py`

### Hypergeometric series helpers and tail bounds
- Core hypergeom and remainders: `src/arbplusjax/hypgeom.py`
- Series helper utilities and gaps: `src/arbplusjax/series_missing_impl.py`, `src/arbplusjax/series_utils.py`

### SciPy / JAX SciPy
- External comparison only; benchmark and reference surfaces may use `scipy.special` and `jax.scipy.special`, but runtime implementation code should not depend on them.

### mpmath
- Point evaluation with `mp.mp.dps`; intervals only with `mp.iv`.

## How to document per-function differences

For a given function:
1. Identify the **backend** (C Arb, JAX basic, JAX rigorous/adaptive, SciPy, JAX SciPy, mpmath).
2. Locate the **primary implementation** file and any tightening helpers.
3. Note the **bounds method** (analytic/derivative/sampling/series remainder).
4. Note any **domain handling** (branch cuts, endpoint sampling).

Example entry structure (use in module docs):

```
Function: acb_hypgeom_u
JAX basic: midpoint eval + outward rounding.
JAX rigorous: regime selection + corner sampling.
JAX adaptive: fixed stencil sampling.
Tightening: asymptotic/series regime switch + endpoint sampling.
Code: src/arbplusjax/hypgeom.py (acb_hypgeom_u, _complex_hypu_regime)
```
