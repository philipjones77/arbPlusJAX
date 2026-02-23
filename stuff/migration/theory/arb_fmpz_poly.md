# arb_fmpz_poly

## Scope

- Cubic polynomial evaluation on real intervals using midpoint evaluation.

## Intended API Surface

- C reference library: `arb_fmpz_poly_ref`
  - `arb_fmpz_poly_eval_cubic_ref(coeffs, x)`
  - Batch variant
- JAX module: `arbjax.arb_fmpz_poly`
  - `arb_fmpz_poly_eval_cubic(coeffs, x)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Coefficients and x evaluated at midpoints.
- Outward rounding applied to final real result.

## Differentiability

- Differentiable w.r.t. coefficients and x on smooth subdomains.

## Notes

- Shape contract: coeffs `(..., 4, 2)` and x `(..., 2)`.
