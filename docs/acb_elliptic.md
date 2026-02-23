# acb_elliptic

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Minimal chassis for complete elliptic integrals K(m) and E(m) via AGM.
- Evaluated at complex midpoint only.

## Intended API Surface

- C reference library: `acb_elliptic_ref`
  - `acb_elliptic_k_ref(m)`
  - `acb_elliptic_e_ref(m)`
  - Batch variants
- JAX module: `arbplusjax.acb_elliptic`
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

## Formulas

- Complete elliptic K(m)=π/2·AGM(1, √(1-m)).
- E(m) computed via AGM-based approximation on midpoint.

## Implementation Notes

- AGM iteration on midpoint; interval widening on non-finite.

