Last updated: 2026-02-25T03:51:38Z

# poly_wrappers

Mode dispatch for polynomial kernels in `arb_poly`, `acb_poly`, and `arb_fmpz_poly`.

## Rigorous handling

Rigorous mode uses interval Horner evaluation for low‑degree kernels where explicit formulas are present (e.g., cubic evaluation). Other functions fall back to Jacobian or adaptive sampling bounds.
