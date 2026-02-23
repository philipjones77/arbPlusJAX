# acb_elliptic

## Scope

- Minimal chassis for complete elliptic integrals K(m) and E(m) via AGM.
- Evaluated at complex midpoint only.

## Intended API Surface

- C reference library: `acb_elliptic_ref`
  - `acb_elliptic_k_ref(m)`
  - `acb_elliptic_e_ref(m)`
  - Batch variants
- JAX module: `arbjax.acb_elliptic`
  - `acb_elliptic_k(m)`
  - `acb_elliptic_e(m)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Uses fixed-iteration AGM (8 steps).
- Outward rounding applied to final complex result.
- If non-finite, return full interval box.

## Differentiability

- Differentiable w.r.t. `m` on smooth subdomains.
- AGM iterations are pure JAX and support `jit` and `grad`.

## Notes

- This is a scaffold, not a full Arb implementation.
