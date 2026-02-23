# fmpz_extras

## Scope

- Small integer helper chassis for fmpz-related operations.

## Implemented Chassis

- `add` and `mul` for `int64`.
- Batch wrappers and JIT.

## Accuracy/Precision Semantics

- Exact within 64-bit integer range; no overflow handling.

## Differentiability

- Not differentiable with respect to integer inputs.

## Notes

- Replace with big-integer semantics when integrating full fmpz.

## Formulas

- int64 add/mul primitives.
