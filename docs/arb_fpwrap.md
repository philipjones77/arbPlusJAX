# arb_fpwrap

## Scope

- Minimal floating wrappers for exp/log on real and complex inputs.

## Intended API Surface

- C reference library: `arb_fpwrap_ref`
  - `arb_fpwrap_double_exp_ref`, `arb_fpwrap_double_log_ref`
  - `arb_fpwrap_cdouble_exp_ref`, `arb_fpwrap_cdouble_log_ref`
- JAX module: `arbplusjax.arb_fpwrap`
  - `arb_fpwrap_double_exp`, `arb_fpwrap_double_log`
  - `arb_fpwrap_cdouble_exp`, `arb_fpwrap_cdouble_log`

## Accuracy/Precision Semantics

- Uses plain double precision operations.
- Return codes: 0 on finite result, 1 on non-finite.

## Differentiability

- Differentiable w.r.t. inputs on smooth subdomains.

## Notes

- This is a minimal scaffold; real Arb fpwrap covers many more functions.

## Formulas

- exp/log wrappers on floating midpoint with outward rounding.

## Implementation Notes

- Intended for compatibility with C fpwrap.
