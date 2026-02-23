# dirichlet

## Scope

- Real interval versions of zeta/eta using naive Dirichlet series.
- Inputs are `double_interval` boxes; outputs are outward-rounded intervals.

## Implemented Chassis

- `zeta(s) = sum_{k=1}^n k^{-s}` on the midpoint `s`.
- `eta(s) = (1 - 2^{1-s}) * zeta(s)` on the midpoint `s`.
- Batch wrappers and precision-rounded variants.

## Accuracy/Precision Semantics

- Evaluate series at midpoint; wrap result with outward rounding via `double_interval`.
- `*_prec` APIs apply `round_interval_outward` to enforce Arb-like precision semantics.
- Non-finite results return full intervals.

## Differentiability

- Differentiable through the midpoint-based scalar path for `s` in the smooth region (typically `s > 1`).

## Notes

- This is a scaffold; replace with Arb-style Euler-Maclaurin / acceleration when upgrading accuracy.
