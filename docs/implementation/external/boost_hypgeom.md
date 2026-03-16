Last updated: 2026-03-13T00:00:00Z

# boost_hypgeom

Boost-derived hypergeometric compatibility layer implemented in pure JAX with the `boost_` prefix.

This module is an external-lineage surface: it preserves Boost-style provenance at the module and compatibility-layer level while still using repo-standard JAX execution modes and interval/box wrappers.

Naming direction:

- the long-term public math surface should prefer the mathematical function name itself
- Boost lineage should stay separable through implementation metadata, registry entries, module lineage, and explicit dispatch controls
- Boost-prefixed callables in this module should be treated as compatibility or implementation-specific entry points, not as the preferred final naming pattern for the canonical public API

Public APIs:

- `boost_hypergeometric_1f0`
- `boost_hypergeometric_0f1`
- `boost_hypergeometric_2f0`
- `boost_hypergeometric_1f1`
- `boost_hypergeometric_2f1`
- `boost_hypergeometric_pfq`
- `boost_hypergeometric_pfq_precision`

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

Batch/helper coverage currently present:

- fixed and padded point helpers for `0f1`, `1f1`, `2f1`-series, and `pfq`
- fixed and padded precision wrappers for the same families
- fixed and padded mode-dispatch helpers for `0f1`, `1f1`, `2f1`-series, and `pfq`

Mode semantics:

- `point`: direct JAX point evaluation
- `basic`: point result boxed/interval-rounded to the requested precision
- `rigorous`: routed through repo-standard rigorous interval/box kernels
- `adaptive`: routed through repo-standard adaptive interval/box kernels

Real vs complex behavior:

- real-valued families use interval helpers from `double_interval`
- complex-valued families use complex-box helpers from `acb_core`
- several public Boost-prefixed APIs delegate rigorous/adaptive behavior to existing canonical `hypgeom` or wrapper kernels rather than reimplementing separate Boost-specific enclosure logic

Current implemented public surface in the code:

- direct point/basic/rigorous/adaptive API wrappers for `1f0`, `0f1`, `2f0`, `1f1`, `2f1`, and generic `pfq`
- explicit precision-selection helper via `boost_hypergeometric_pfq_precision`
- provenance-prefixed aliases for Boost-inspired internal regimes (`1f1` series/asymptotic, `2f1` series/continued-fraction/Pade/rational, `1f2` series)

Design notes:

- Thin Python dispatch only at API entry.
- JAX-first kernels for point evaluation.
- Interval and box modes routed through existing arb/acb kernels and mode wrappers.
- Source inventory and provenance are documented in `boost_math_hypgeom_inventory.md`.

Current backlog:

- migrate toward canonical public names with implementation selection instead of relying on `boost_` in the main user-facing name
- broaden batch coverage from the current `0f1` / `1f1` / `2f1`-series / `pfq` subset to the full public Boost-prefixed surface
- tighten family-specific rigorous/adaptive kernels where the current implementation still relies on generic inflators or canonical-family delegation
- document which public Boost APIs correspond to true Boost top-level functions versus Boost-detail-inspired helper surfaces
