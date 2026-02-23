# acb_mat

## Scope

- 2x2 determinant and trace on complex interval matrices using midpoint evaluation.

## Intended API Surface

- C reference library: `acb_mat_ref`
  - `acb_mat_2x2_det_ref(a)`
  - `acb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbjax.acb_mat`
  - `acb_mat_2x2_det(a)`
  - `acb_mat_2x2_trace(a)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Matrix entries interpreted as `acb_box_t`.
- Computation uses midpoint complex matrix then outward rounds.

## Differentiability

- Differentiable w.r.t. matrix entries for smooth subdomains.

## Notes

- Shape contract: `(..., 2, 2, 4)` for matrices.
