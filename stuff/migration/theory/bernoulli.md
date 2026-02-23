# bernoulli

## Scope

- Fixed small Bernoulli numbers for n in {0,1,2,4}.

## Intended API Surface

- C reference library: `bernoulli_ref`
  - `bernoulli_number_ref(n)`
  - Batch variant
- JAX module: `arbjax.bernoulli`
  - `bernoulli_number(n)`
  - Batch/jit variants

## Accuracy/Precision Semantics

- Hard-coded exact values in double precision.

## Differentiability

- Returns constants; gradients are zero.

## Notes

- Scaffold only; not a full Bernoulli implementation.
