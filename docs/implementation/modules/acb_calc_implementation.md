Last updated: 2026-02-25T03:51:38Z

# acb_calc

## Precision Modes

- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="basic|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Straight-line complex line integration for unary integrands.
- Supported integrands include elementary and selected special functions:
  `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
  `log1p`, `expm1`, `sin_pi`, `cos_pi`, `tan_pi`, `sinc`, `sinc_pi`,
  `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`,
  `gamma`, `erf`, `erfc`, `erfi`, `barnesg`.
- Straight-line path from `a` to `b` using interval-box midpoints.

## Method Families vs Mode Dispatch

`acb_calc` contains several integration method families:

- `acb_calc_integrate_line`: midpoint-rule line integration
- `acb_calc_integrate_gl_auto_deg`: Gauss-Legendre line integration with automatic degree refinement
- `acb_calc_integrate_taylor`: midpoint-centered Taylor-series line integration

These are not themselves the four arbPlusJAX modes.

The four modes are applied one layer above through `calc_wrappers` and the public API:

- `point`: point-only evaluation path where available
- `basic`: direct method result with outward rounding
- `adaptive`: wrapper-level adaptive tightening around the kernel
- `rigorous`: wrapper-level rigorous enclosure / method-specific rigorous kernel

So `gl_auto_deg` and `taylor` are distinct calc methods, while `point|basic|adaptive|rigorous`
are dispatch modes applied to those methods.

## Intended API Surface

- C reference library: `acb_calc_ref`
  - `acb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - `acb_calc_integrate_line_batch_ref(...)`
- JAX module: `arbplusjax.acb_calc`
  - `acb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - `acb_calc_integrate_line_prec(...)`
  - Batch + JIT variants

## Accuracy/Precision Semantics

- basic integrates along midpoints only (not full enclosure).
- Rigorous uses interval evaluation and tightens with a coarse/fine intersection.
- Outward rounding applied to the final complex result.
- If non-finite, return full interval box.

## Differentiability

- Differentiable w.r.t. `a` and `b` on smooth subdomains.
- `integrand` and `n_steps` are static JIT args.

## Notes

- This is a chassis-level approximation; it does not replicate Arb's adaptive integration.
- The C parity layer only covers the legacy `exp/sin/cos` subset for `acb_calc_integrate_line_ref`.
- `gl_auto_deg` and `taylor` are method names, not mode names.

## Formulas

- Line integral along segment: z(t)=a+(b-a)t, t∈[0,1].
- Integral approximation: ∫_0^1 f(z(t)) (b-a) dt ≈ (b-a)/N ∑ f(z(t_i)).

## Implementation Notes

- `acb_calc_integrate_line` uses midpoint sampling over `N` steps on the complex box midpoint.
- `acb_calc_integrate_gl_auto_deg` uses Gauss-Legendre nodes/weights with auto-selected degree refinement.
- `acb_calc_integrate_taylor` uses a midpoint-centered Taylor expansion and truncation-based enclosure.
- Returns full boxes on non-finite values.
