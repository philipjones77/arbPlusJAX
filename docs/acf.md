# acf

## Scope

- Basic complex float add and multiply.

## Intended API Surface

- C reference library: `acf_ref`
  - `acf_add_ref(a, b)`
  - `acf_mul_ref(a, b)`
  - Batch variants
- JAX module: `arbplusjax.acf`
  - `acf_add(a, b)`
  - `acf_mul(a, b)`
  - Batch/jit variants

## Accuracy/Precision Semantics

- Uses complex64/complex128 arithmetic depending on inputs (default complex128).

## Differentiability

- Differentiable w.r.t. complex inputs (real/imag components).

## Notes

- Not interval arithmetic; intended as a simple scaffold.

## Formulas

- Complex add/mul on midpoints.

## Implementation Notes

- Complex float (no interval) with JAX complex128.
