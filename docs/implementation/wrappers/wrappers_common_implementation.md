Last updated: 2026-02-25T03:51:38Z

# wrappers_common

Shared interval/box bound logic for wrappers.

## Jacobian bound

For a midpoint kernel \(g\), with midpoint \(m\) and radii \(r\):
\[
|g(x) - g(m)| \le |J(m)| r
\]
This is used for rigorous bounds when analytic interval formulas are unavailable.

## Adaptive bound

Sampling at \(\{m, m+r, m-r\}\) produces a non‑rigorous but usually tighter bound.

## Helpers

- `resolve_prec_bits` selects precision based on `prec_bits` or `dps`.
- `inflate_interval` and `inflate_acb` expand bounds to account for precision.
