# Precision and Interval Semantics

## Summary

arbPlusJAX is JAX-first and uses **float64 interval boxes** with outward rounding. Precision control is modeled by widening bounds after computation. This differs from Arb’s **ball arithmetic**, which tracks midpoint and radius with arbitrary precision through every operation.

We standardize **three modes** across baseline modules:

- **baseline**: midpoint evaluation + outward rounding (fastest, least rigorous)
- **rigorous**: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint
- **adaptive**: fixed-grid sampling around the midpoint with extra inflation (non‑recompilable heuristic)

## Precision comparison (mpmath vs JAX vs Arb vs jax.special)

| System | Precision model | How to set | Notes |
|---|---|---|---|
| mpmath | Arbitrary precision (decimal/bits) | `mp.mp.dps`, `mp.mp.prec` | Real/complex high precision; interval mode via `mp.iv`. |
| arbPlusJAX | Fixed float64/complex128 | `prec_bits` (outward rounding only) | No true arbitrary precision; `prec_bits` widens bounds post‑hoc. |
| Arb (C) | Arbitrary precision ball arithmetic | `prec` (bits) per call | Rigorous midpoint + radius propagation. |
| jax.special | Fixed float64/complex128 | dtype (float32/64) | Same numeric precision as JAX; no interval/ball semantics. |

## mpmath-style interface in arbPlusJAX

For JAX compatibility we provide a **mpmath-like API** that maps `dps`/`prec` to `prec_bits` and only **inflates bounds** after float64 midpoint evaluation:

- `precision.set_dps(dps)` / `precision.set_prec_bits(prec_bits)`
- `precision.workdps(dps)` / `precision.workprec(prec_bits)`
- `ball_wrappers.*_mp(...)` helpers for `exp`, `log`, `sin`, `gamma`

This does **not** change the underlying arithmetic precision; it only changes the outward rounding inflation.

## Methodologies with math

### 1) Baseline JAX interval kernels (arbPlusJAX)

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

Same return box as above. This is a heuristic but often tighter than baseline for nonlinear functions.

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

### 6) jax.special

`jax.scipy.special` functions are standard floating‑point evaluations with dtype‑level precision (float32/float64). No enclosure, no error bounds.

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

## Three-mode semantics (current standard)

### Baseline

For any `*_prec` kernel we compute midpoint output and apply outward rounding:
\[
y = f(m), \quad \mathrm{out} = \mathrm{round\_outward}(y, \text{prec\_bits})
\]

### Rigorous (analytic where available)

For core functions with known identities (exp/log/sin/cos/tan/sinh/cosh/tanh, sqrt):

- Real: monotonicity + critical point checks.
- Complex: analytic formulas (e.g. \(\exp(x+iy)=e^x(\cos y+i\sin y)\)) with interval propagation.
- Branch cuts handled by widening (e.g. `log` imag part widened to \([-\pi,\pi]\) if the box crosses the negative real axis or contains 0).
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

### Adaptive (sampling)

We evaluate the midpoint kernel at a small fixed stencil:
\[
\{m,\ m+r,\ m-r\}
\]
and set:
\[
\mathrm{rad}_{\text{out}} = \max_i |g(x_i)-g(m)| + 2^{-\text{prec\_bits}}
\]

This is not fully rigorous but usually tighter than baseline in nonlinear regions. It is static‑shape and JIT‑friendly.

### Where this is implemented

- **Core (analytic bounds)**: `arb_core`, `acb_core` via `core_wrappers`.
- **Generic bound tightening**: `dirichlet_wrappers`, `calc_wrappers`, `mat_wrappers`, `poly_wrappers`, `modular_elliptic_wrappers` use Jacobian/sampling bounds.
- **DFT/Convolution (analytic interval arithmetic)**: `dft_wrappers` uses explicit interval twiddle factors and interval linear algebra in rigorous mode.
- **Ball wrappers**: `ball_wrappers` for exp/log/sin/gamma with derivative/sampling bounds.
