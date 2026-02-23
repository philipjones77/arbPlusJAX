# fmpzi

## Scope

- Integer interval (fmpzi) chassis.

## Implemented Chassis

- `interval(lo, hi)` with ordering enforcement.
- `add` and `sub` interval arithmetic.
- Batch wrappers and JIT.

## Accuracy/Precision Semantics

- Exact within 64-bit integer range; no overflow handling.

## Differentiability

- Not differentiable with respect to integer inputs.

## Notes

- Extend with multiplication and gcd when full semantics are required.
