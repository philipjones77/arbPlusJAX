# bool_mat

## Scope

- Boolean 2x2 matrix primitives over GF(2).
- Flattened storage order: `[a00, a01, a10, a11]`.

## Implemented Chassis

- Determinant: `det = a00*a11 + a01*a10 (mod 2)`.
- Trace: `trace = a00 + a11 (mod 2)`.

## Accuracy/Precision Semantics

- Exact in GF(2) via bitwise AND/XOR on `uint8`.
- No floating rounding concerns.

## Differentiability

- Not differentiable w.r.t. inputs; gradients are only defined for unrelated parameters.

## Notes

- Extend to larger matrices with bit-packed SIMD once correctness tests stabilize.
