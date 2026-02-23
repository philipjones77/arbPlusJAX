# dft_wrappers

Mode dispatch for DFT and convolution kernels.

## Rigorous handling

Rigorous mode uses analytic interval twiddle factors and interval arithmetic (`acb_dft_*_rigorous`, `acb_convol_*_rigorous`) instead of generic Jacobian bounds.

## Adaptive handling

Adaptive mode uses sampling bounds on midpoint kernels for a cheaper, non‑rigorous interval.
