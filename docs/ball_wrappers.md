# Ball Wrappers (Rigorous vs Adaptive)

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

This module introduces two interval strategies for `exp`, `log`, `sin`, and `gamma`.

## Rigorous (fixed‑eps) wrapper

- Represent each interval/box as a **ball**: midpoint + radius.
- Propagate radius using a **sampled derivative bound**:
  - Real: `rad_out = max |f'(mid + r t_i)| * rad_in + eps`
  - Complex: `rad_out = max ||J(mid + r e^{iθ_i})||_F * rad_in + eps`
- `eps = 2^{-prec_bits}` is fixed and does not change loop structure.

This is **deterministic and differentiable**, but still a heuristic compared to Arb.

## Adaptive (non‑recompilable) wrapper

- Keep a fixed sample grid (static size).
- Evaluate `f` at sample points around the midpoint.
- Set `rad_out` from the maximum deviation: `max |f(z_i) - f(mid)| + eps`.

This avoids recompilation because the grid size is static, but can be looser than the derivative bound.

## API

Real:

- `arb_ball_exp`, `arb_ball_log`, `arb_ball_sin`, `arb_ball_gamma`
- `arb_ball_*_adaptive`

Complex:

- `acb_ball_exp`, `acb_ball_log`, `acb_ball_sin`, `acb_ball_gamma`
- `acb_ball_*_adaptive`

## Mode selection wrapper

For convenience, see `baseline_wrappers.*_mp` which lets you select:

- `mode="baseline" | "rigorous" | "adaptive"`
- `dps` or `prec_bits`

## Comparison

Use `tools/compare_ball_wrappers.py` to compare:

- baseline JAX interval wrappers
- rigorous ball wrappers
- adaptive ball wrappers
- C reference (when available)

