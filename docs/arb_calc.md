# arb_calc

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Midpoint-rule line integration for real intervals and integrands `exp`, `sin`, `cos`.

## Intended API Surface

- C reference library: `arb_calc_ref`
  - `arb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - Batch variant
- JAX module: `arbplusjax.arb_calc`
  - `arb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Baseline integrates midpoint path only (not full enclosure).
- Rigorous uses interval evaluation and tightens with a coarse/fine intersection.
- Outward rounding applied to final real result.
- If non-finite, return full interval.

## Differentiability

- Differentiable w.r.t. `a` and `b` on smooth subdomains.

## Notes

- This is a chassis approximation; it does not replicate Arb's adaptive integration.

## Formulas

- Real line integral: ∫_0^1 f(a+(b-a)t) (b-a) dt.

## Implementation Notes

- Midpoint sampling and outward rounding.

