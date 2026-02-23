# arb_calc

## Scope

- Midpoint-rule line integration for real intervals and integrands `exp`, `sin`, `cos`.

## Intended API Surface

- C reference library: `arb_calc_ref`
  - `arb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - Batch variant
- JAX module: `arbjax.arb_calc`
  - `arb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Integrates midpoint path only (not full enclosure).
- Outward rounding applied to final real result.
- If non-finite, return full interval.

## Differentiability

- Differentiable w.r.t. `a` and `b` on smooth subdomains.

## Notes

- This is a chassis approximation; it does not replicate Arb's adaptive integration.
