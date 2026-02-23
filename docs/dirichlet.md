# dirichlet

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Real interval versions of zeta/eta using naive Dirichlet series.
- Inputs are `double_interval` boxes; outputs are outward-rounded intervals.

## Implemented Chassis

- `zeta(s) = sum_{k=1}^n k^{-s}` on the midpoint `s`.
- `eta(s) = (1 - 2^{1-s}) * zeta(s)` on the midpoint `s`.
- Batch wrappers and precision-rounded variants.

## Accuracy/Precision Semantics

- Baseline: evaluate series at midpoint; wrap result with outward rounding via `double_interval`.
- Rigorous: interval series with explicit remainder bounds:
  - ζ tail bound via integral test when `Re(s)>1`.
  - η tail bound via alternating series remainder when `Re(s)>0`.
- `*_prec` APIs apply `round_interval_outward` to enforce Arb-like precision semantics.
- Non-finite or invalid parameter ranges return full intervals.

## Differentiability

- Differentiable through the midpoint-based scalar path for `s` in the smooth region (typically `s > 1`).

## Notes

- This is a scaffold; replace with Arb-style Euler-Maclaurin / acceleration when upgrading accuracy.

## Formulas

- ζ(s)=∑_{n=1}^N n^{-s}, η(s)=∑_{n=1}^N (-1)^{n-1} n^{-s}.

