Last updated: 2026-03-01T00:00:00Z

# cusf_compat

This module documents the `Cusf_*` compatibility layer implemented in:

- `src/arbplusjax/cusf_compat.py`

## Provenance (separate implementation lineage)

`Cusf_*` functions are a **separate compatibility implementation track** from the main Arb/ACB-style naming in `arbPlusJAX`.

- External source lineage: `cusf-master` (CUDA Special Functions project)
  - Path inspected: `C:\Users\phili\OneDrive - University of Arizona\Documents\library\cusf-master\cusf-master`
  - Reference status table source: `README.md` in that repo
- In this project, the compatibility layer is implemented in **pure JAX** with the prefix:
  - `Cusf_<function_name>`

This keeps CUSF-derived API names explicit and separate from native `arb_*` / `acb_*` interfaces.

## Mode model

`Cusf_*` APIs support four modes where applicable:

- `point`: scalar/array point evaluation (no interval enclosure)
- `basic`: interval from point path + outward rounding
- `rigorous`: interval mode delegated to existing rigorous wrappers
- `adaptive`: interval mode delegated to existing adaptive wrappers

Mode keyword:

- `mode="point|basic|rigorous|adaptive"`

Precision keyword:

- `prec_bits` for interval modes

## Function mapping

The following CUSF-style functions are exposed:

- `Cusf_Hyp2f1`
- `Cusf_Hyp1f1`
- `Cusf_faddeeva_w`
- `Cusf_erf`
- `Cusf_jy`
- `Cusf_kv`
- `Cusf_iv`
- `Cusf_j0`
- `Cusf_yv`
- `Cusf_jvp`
- `Cusf_y0`
- `Cusf_jv`
- `Cusf_k0`
- `Cusf_kvp`
- `Cusf_i0`
- `Cusf_ivp`
- `Cusf_yvp`

## Helper mapping

CUSF helper-style names exposed in this compatibility module:

- `Cusf_digamma`
- `Cusf_gamma`
- `Cusf_tgamma1pmv`
- `Cusf_chebyshev`
- `Cusf_polynomial`
- `Cusf_poly_rational`

## Internal backend usage

The compatibility functions reuse existing JAX kernels in `arbPlusJAX` where possible:

- Hypergeometric and Bessel point kernels from `hypgeom.py`
- Interval mode dispatch from `baseline_wrappers.py` / `hypgeom_wrappers.py`
- Interval arithmetic from `double_interval.py`

This means the `Cusf_*` layer is maintained as a naming/API compatibility surface, not a duplicate numerical backend.

## Notes

- `Cusf_faddeeva_w` currently uses the project's complex `erfc` series path in pure JAX.
- For derivatives (`jvp`, `yvp`, `ivp`, `kvp`), point mode uses standard recurrence identities; interval modes apply the same identities over interval outputs.
