# dirichlet_wrappers

Mode dispatch for Dirichlet functions.

## Modes

- `baseline`: midpoint + outward rounding.
- `rigorous`: uses analytic series bounds where implemented (zeta, eta) and Jacobian bounds otherwise.
- `adaptive`: sampling‑based bound.

This wrapper ensures consistent precision handling across `dirichlet` functions.
