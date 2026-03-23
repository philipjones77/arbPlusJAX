Last updated: 2026-02-25T03:51:38Z

# arb_calc

## Precision Modes

- `basic`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="basic|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Straight-line real line integration for unary integrands.
- Supported integrands include:
  `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
  `log1p`, `expm1`, `sin_pi`, `cos_pi`, `tan_pi`, `sinc`, `sinc_pi`,
  `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`, `cbrt`,
  `gamma`, `erf`, `erfc`, `erfi`, `barnesg`.

## Method Families vs Mode Dispatch

`arb_calc` primarily exposes a midpoint-rule line-integration family:

- `arb_calc_integrate_line`
- `arb_calc_integrate_line_rigorous`
- batch / precision variants

These are calc kernels, not the full four-mode dispatch surface by themselves.

The standard arbPlusJAX modes are applied above this layer through `calc_wrappers` and `api`:

- `point`
- `basic`
- `adaptive`
- `rigorous`

That distinction matters when comparing modules: calc function names such as
`integrate_line`, `gl_auto_deg`, or `taylor` describe numerical method families,
while mode names describe how arbPlusJAX dispatches and tightens those kernels.

## Intended API Surface

- C reference library: `arb_calc_ref`
  - `arb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - Batch variant
- JAX module: `arbplusjax.arb_calc`
  - `arb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- basic integrates midpoint path only (not full enclosure).
- Rigorous uses interval evaluation and tightens with a coarse/fine intersection.
- Outward rounding applied to final real result.
- If non-finite, return full interval.

## Differentiability

- Differentiable w.r.t. `a` and `b` on smooth subdomains.

## Notes

- This is a chassis approximation; it does not replicate Arb's adaptive integration.
- The C parity layer currently covers the legacy `exp/sin/cos` subset for `arb_calc_integrate_line_ref`.

## Formulas

- Real line integral: ∫_0^1 f(a+(b-a)t) (b-a) dt.

## Implementation Notes

- Midpoint sampling and outward rounding.
