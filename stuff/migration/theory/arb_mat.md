# arb_mat

## Scope

- 2x2 determinant and trace on real interval matrices using midpoint evaluation.

## Intended API Surface

- C reference library: `arb_mat_ref`
  - `arb_mat_2x2_det_ref(a)`
  - `arb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbjax.arb_mat`
  - `arb_mat_2x2_det(a)`
  - `arb_mat_2x2_trace(a)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Matrix entries interpreted as intervals.
- Computation uses midpoint matrix then outward rounds.

## Differentiability

- Differentiable w.r.t. matrix entries on smooth subdomains.

## Notes

- Shape contract: `(..., 2, 2, 2)` for matrices.
