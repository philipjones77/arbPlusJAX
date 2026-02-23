# acb_poly

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Cubic polynomial evaluation on complex interval inputs using midpoint evaluation.

## Intended API Surface

- C reference library: `acb_poly_ref`
  - `acb_poly_eval_cubic_ref(coeffs, z)`
  - Batch variant
- JAX module: `arbplusjax.acb_poly`
  - `acb_poly_eval_cubic(coeffs, z)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Uses midpoint for coefficients and z.
- Outward rounding applied to final complex result.

## Differentiability

- Differentiable w.r.t. coefficients and z for smooth subdomains.

## Notes

- Shape contract: coeffs `(..., 4, 4)` and z `(..., 4)`.

## Formulas

- Cubic evaluation: p(x)=a0+a1 x+a2 x^2+a3 x^3.

## Implementation Notes

- Horner-style on midpoint.

