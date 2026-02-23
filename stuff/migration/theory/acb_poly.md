# acb_poly

## Scope

- Cubic polynomial evaluation on complex interval inputs using midpoint evaluation.

## Intended API Surface

- C reference library: `acb_poly_ref`
  - `acb_poly_eval_cubic_ref(coeffs, z)`
  - Batch variant
- JAX module: `arbjax.acb_poly`
  - `acb_poly_eval_cubic(coeffs, z)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Uses midpoint for coefficients and z.
- Outward rounding applied to final complex result.

## Differentiability

- Differentiable w.r.t. coefficients and z for smooth subdomains.

## Notes

- Shape contract: coeffs `(..., 4, 4)` and z `(..., 4)`.
