Last updated: 2026-02-25T03:51:38Z

# core_wrappers

Mode dispatch for `arb_core` and `acb_core` kernels.

## Modes

- `basic`: midpoint + outward rounding to `prec_bits`.
- `rigorous`: analytic interval formulas (exp/log/sin/cos/tan/sinh/cosh/tanh, sqrt).
- `adaptive`: sampling or Jacobian bounds when analytic bounds are not available.

## Notes

Complex `log` widens the imaginary interval to cover branch cuts when the box crosses the negative real axis or contains zero. Complex `sqrt` uses a polar form with interval theta.
