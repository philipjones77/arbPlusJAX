# mag

## Scope

- Magnitude (mag) chassis for non-negative reals.

## Implemented Chassis

- `add` and `mul` on `float64`.
- Batch wrappers and JIT.

## Accuracy/Precision Semantics

- Double-precision arithmetic; no explicit rounding control.

## Differentiability

- Fully differentiable for smooth inputs.

## Notes

- Extend with normalization and bound propagation for Arb-style mag types.

## Formulas

- Magnitude add/mul as float boxes.
