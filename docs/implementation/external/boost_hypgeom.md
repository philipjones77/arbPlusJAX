Last updated: 2026-03-01T00:00:00Z

# boost_hypgeom

Boost-derived hypergeometric compatibility layer implemented in pure JAX with the `boost_` prefix.

Public APIs:

- `boost_hypergeometric_1F0`
- `boost_hypergeometric_0F1`
- `boost_hypergeometric_2F0`
- `boost_hypergeometric_1F1`
- `boost_hypergeometric_pFq`
- `boost_hypergeometric_pFq_precision`

Each API supports the four modes:

- `point`
- `basic`
- `rigorous`
- `adaptive`

Internal/helper APIs (also prefixed):

- `boost_hyp1f1_series`
- `boost_hyp1f1_asym`
- `boost_hyp2f1_series`
- `boost_hyp2f1_cf`
- `boost_hyp2f1_pade`
- `boost_hyp2f1_rational`
- `boost_hyp1f2_series`

Design notes:

- Thin Python dispatch only at API entry.
- JAX-first kernels for point evaluation.
- Interval and box modes routed through existing arb/acb kernels and mode wrappers.
- Source inventory and provenance are documented in `boost_math_hypgeom_inventory.md`.
