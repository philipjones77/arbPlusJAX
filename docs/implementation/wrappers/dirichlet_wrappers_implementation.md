Last updated: 2026-02-25T03:51:38Z

# dirichlet_wrappers

Mode dispatch for Dirichlet functions.

## Modes

- `basic`: midpoint + outward rounding.
- `rigorous`: uses analytic series bounds where implemented (zeta, eta) and Jacobian bounds otherwise.
- `adaptive`: sampling‑based bound.

This wrapper ensures consistent precision handling across `dirichlet` functions.
