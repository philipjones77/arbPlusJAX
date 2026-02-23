# acb_calc

## Scope

- Midpoint-rule line integration for a small set of integrands (`exp`, `sin`, `cos`).
- Straight-line path from `a` to `b` using interval-box midpoints.

## Intended API Surface

- C reference library: `acb_calc_ref`
  - `acb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - `acb_calc_integrate_line_batch_ref(...)`
- JAX module: `arbjax.acb_calc`
  - `acb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - `acb_calc_integrate_line_prec(...)`
  - Batch + JIT variants

## Accuracy/Precision Semantics

- Integrate along midpoints only (not full enclosure).
- Outward rounding applied to the final complex result.
- If non-finite, return full interval box.

## Differentiability

- Differentiable w.r.t. `a` and `b` on smooth subdomains.
- `integrand` and `n_steps` are static JIT args.

## Notes

- This is a chassis-level approximation; it does not replicate Arb's adaptive integration.
- Integrand IDs:
  - `0`: exp
  - `1`: sin
  - `2`: cos
