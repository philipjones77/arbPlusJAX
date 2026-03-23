Last updated: 2026-02-25T03:51:38Z

# mat_wrappers

Mode dispatch for matrix kernels in `arb_mat` and `acb_mat`.

## Modes

- `point`: first-class midpoint-only path that dispatches to dedicated matrix point kernels in `point_wrappers.py`
- `basic`: interval/box kernel with outward rounding
- `rigorous`: specialized exact/tight formulas where available, else Jacobian-style enclosure
- `adaptive`: sampled tightening around the midpoint

Point mode is intentionally separate from interval kernels so JAX can compile and run it as a pure point-valued linear-algebra path.

## Rigorous handling

Rigorous mode uses explicit interval formulas where provided (2x2 determinant and trace). Other functions fall back to Jacobian or adaptive sampling bounds.
