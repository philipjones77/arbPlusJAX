# arb_poly

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Cubic polynomial evaluation on real intervals using midpoint evaluation.

## Intended API Surface

- C reference library: `arb_poly_ref`
  - `arb_poly_eval_cubic_ref(coeffs, x)`
  - Batch variant
- JAX module: `arbplusjax.arb_poly`
  - `arb_poly_eval_cubic(coeffs, x)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Coefficients and x evaluated at midpoints.
- Outward rounding applied to final real result.

## Differentiability

- Differentiable w.r.t. coefficients and x on smooth subdomains.

## Notes

- Shape contract: coeffs `(..., 4, 2)` and x `(..., 2)`.

## Formulas

- Cubic evaluation: p(x)=a0+a1 x+a2 x^2+a3 x^3.

