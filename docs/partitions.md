# partitions

## Scope

- Integer partition function chassis.

## Implemented Chassis

- `p(n)` via Euler pentagonal recurrence.
- Batch wrapper and JIT.

## Accuracy/Precision Semantics

- Exact within 64-bit integer range; no overflow handling.

## Differentiability

- Not differentiable with respect to integer inputs.

## Notes

- Replace with Arb/FLINT partition algorithms for high `n`.

## Formulas

- Partition numbers via Euler pentagonal recurrence.
