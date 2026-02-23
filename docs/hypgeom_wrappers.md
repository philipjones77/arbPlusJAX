# hypgeom_wrappers

Mode dispatch for hypergeometric and special functions in `hypgeom`.

## Handling

Baseline uses midpoint + outward rounding. Rigorous mode uses ball wrappers where available (gamma, log, exp, sin) and Jacobian/sampling bounds otherwise. Adaptive mode uses sampling bounds.
