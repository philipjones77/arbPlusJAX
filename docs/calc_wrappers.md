# calc_wrappers

Mode dispatch for `arb_calc` and `acb_calc`.

## Integration bounds

Rigorous mode uses interval evaluation of the midpoint quadrature kernel over the integration interval. Adaptive mode uses sampling, and baseline is midpoint with outward rounding.
