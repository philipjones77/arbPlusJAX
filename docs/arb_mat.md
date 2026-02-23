# arb_mat

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- 2x2 determinant and trace on real interval matrices using midpoint evaluation.

## Intended API Surface

- C reference library: `arb_mat_ref`
  - `arb_mat_2x2_det_ref(a)`
  - `arb_mat_2x2_trace_ref(a)`
  - Batch variants
- JAX module: `arbplusjax.arb_mat`
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

## Formulas

- det([[a,b],[c,d]])=ad-bc.
- trace([[a,b],[c,d]])=a+d.

