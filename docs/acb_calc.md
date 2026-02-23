# acb_calc

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Midpoint-rule line integration for a small set of integrands (`exp`, `sin`, `cos`).
- Straight-line path from `a` to `b` using interval-box midpoints.

## Intended API Surface

- C reference library: `acb_calc_ref`
  - `acb_calc_integrate_line_ref(a, b, integrand_id, n_steps)`
  - `acb_calc_integrate_line_batch_ref(...)`
- JAX module: `arbplusjax.acb_calc`
  - `acb_calc_integrate_line(a, b, integrand="exp", n_steps=64)`
  - `acb_calc_integrate_line_prec(...)`
  - Batch + JIT variants

## Accuracy/Precision Semantics

- Baseline integrates along midpoints only (not full enclosure).
- Rigorous uses interval evaluation and tightens with a coarse/fine intersection.
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

## Formulas

- Line integral along segment: z(t)=a+(b-a)t, t∈[0,1].
- Integral approximation: ∫_0^1 f(z(t)) (b-a) dt ≈ (b-a)/N ∑ f(z(t_i)).

## Implementation Notes

- Uses midpoint sampling over N steps on the complex box midpoint.
- Returns full boxes on non-finite values.

