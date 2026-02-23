# acb_dirichlet

## Scope

- Naive Dirichlet zeta and eta series evaluated at interval midpoints.
- Intended only as a chassis scaffold (not Arb's full algorithms).

## Intended API Surface

- C reference library: `acb_dirichlet_ref`
  - `acb_dirichlet_zeta_ref(s, n_terms)`
  - `acb_dirichlet_eta_ref(s, n_terms)`
  - Batch variants
- JAX module: `arbjax.acb_dirichlet`
  - `acb_dirichlet_zeta(s, n_terms=64)`
  - `acb_dirichlet_eta(s, n_terms=64)`
  - Precision/batch/jit variants

## Accuracy/Precision Semantics

- Series evaluated at complex midpoint only.
- Outward rounding applied to the final complex result.
- If non-finite, return full interval box.

## Differentiability

- Differentiable w.r.t. `s` for smooth subdomains.
- `n_terms` is a static JIT arg.

## Notes

- Not valid for all `s` due to slow/divergent series; use only as a migration scaffold.
