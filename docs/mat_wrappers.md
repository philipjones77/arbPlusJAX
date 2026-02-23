# mat_wrappers

Mode dispatch for matrix kernels in `arb_mat` and `acb_mat`.

## Rigorous handling

Rigorous mode uses explicit interval formulas where provided (2x2 determinant and trace). Other functions fall back to Jacobian or adaptive sampling bounds.
