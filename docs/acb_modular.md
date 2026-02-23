# acb_modular

## Precision Modes

- `baseline`: midpoint evaluation + outward rounding (`*_prec`).
- `rigorous`: analytic bounds when available; otherwise Jacobian/Lipschitz bounds around the midpoint.
- `adaptive`: fixed-grid sampling around the midpoint with extra inflation.

Implementation lives in the corresponding `*_wrappers.py` module and uses `impl="baseline|rigorous|adaptive"` plus `dps`/`prec_bits`.

## Scope

- Minimal j-invariant approximation via short q-series.
- Uses complex midpoint for tau.

## Intended API Surface

- C reference library: `acb_modular_ref`
  - `acb_modular_j_ref(tau)`
  - Batch variant
- JAX module: `arbplusjax.acb_modular`
  - `acb_modular_j(tau)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Uses q = exp(2π i τ) and terms: q^{-1} + 744 + 196884 q + 21493760 q^2.
- Outward rounding applied to final complex result.
- If non-finite or q=0, return full interval box.

## Differentiability

- Differentiable w.r.t. `tau` on smooth subdomains.

## Notes

- This is a chassis approximation, not a full modular function implementation.

## Formulas

- j(τ)=q^{-1}+744+196884 q+21493760 q^2+… , q=e^{2πiτ}.

## Implementation Notes

- Truncated q-series on midpoint; full box if non-finite.

