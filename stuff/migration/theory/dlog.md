# dlog

## Scope

- Minimal numeric placeholder for discrete-log module migration.

## Implemented Chassis

- `log1p(x)` on `float64` inputs.
- Batch wrapper and JIT.

## Accuracy/Precision Semantics

- Delegates to `log1p` in libm/JAX (double precision).

## Differentiability

- Fully differentiable for `x > -1`.

## Notes

- Arb's discrete log algorithms are not implemented yet.
