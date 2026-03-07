Last updated: 2026-02-25T03:51:38Z

# calc_wrappers

Mode dispatch for `arb_calc` and `acb_calc`.

## Layering

`calc_wrappers` is the layer that turns calc kernels into the standard arbPlusJAX
mode interface:

- `point`
- `basic`
- `adaptive`
- `rigorous`

This is separate from the numerical method chosen by the calc module itself.

Examples:

- `acb_calc_integrate_line` is a midpoint-rule method
- `acb_calc_integrate_gl_auto_deg` is a Gauss-Legendre method
- `acb_calc_integrate_taylor` is a Taylor-series method

Each of those methods can then be wrapped by `*_mode` entry points generated here.

## Integration bounds

Rigorous mode uses interval evaluation of the midpoint quadrature kernel over the integration interval. Adaptive mode uses sampling, and basic is midpoint with outward rounding.
