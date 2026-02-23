# acb_mat

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- 2x2 determinant and trace on complex interval matrices using midpoint evaluation.

## Intended API Surface

- C reference library: `acb_mat_ref`
  - `acb_mat_2x2_det_ref(a)`
  - `acb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.acb_mat`
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

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.

## Implementation Notes

- Uses complex-box arithmetic on entries.

