# arf

## Scope

- Basic float add and multiply.

## Intended API Surface

- C reference library: `arf_ref`
  - `arf_add_ref(a, b)`
  - `arf_mul_ref(a, b)`
  - Batch variants
- JAX module: `arbjax.arf`
  - `arf_add(a, b)`
  - `arf_mul(a, b)`
  - Batch/jit variants

## Accuracy/Precision Semantics

- Uses double precision arithmetic.

## Differentiability

- Differentiable w.r.t. inputs on smooth subdomains.

## Notes

- Not interval arithmetic; intended as a minimal scaffold.
