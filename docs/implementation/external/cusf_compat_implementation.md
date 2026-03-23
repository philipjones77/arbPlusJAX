Last updated: 2026-03-01T00:00:00Z

# cusf_compat

This module documents the `cusf_*` compatibility layer implemented in:

- `src/arbplusjax/cusf_compat.py`

## Provenance (separate implementation lineage)

`cusf_*` functions are a **separate compatibility implementation track** from the main Arb/ACB-style naming in `arbPlusJAX`.

- External source lineage: `cusf-master` (CUDA Special Functions project)
  - Path inspected: `C:\Users\phili\OneDrive - University of Arizona\Documents\library\cusf-master\cusf-master`
  - Reference status table source: `README.md` in that repo
- In this project, the compatibility layer is implemented in **pure JAX** with the prefix:
  - `cusf_<function_name>`

This keeps CUSF-derived API names explicit and separate from native `arb_*` / `acb_*` interfaces.

## Mode model

`cusf_*` APIs support four modes where applicable:

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

- `cusf_hyp2f1`
- `cusf_hyp1f1`
- `cusf_faddeeva_w`
- `cusf_erf`
- `cusf_besseljy`
- `cusf_besselk`
- `cusf_besseli`
- `cusf_besselj0`
- `cusf_bessely`
- `cusf_besselj_deriv`
- `cusf_bessely0`
- `cusf_besselj`
- `cusf_besselk0`
- `cusf_besselk_deriv`
- `cusf_besseli0`
- `cusf_besseli_deriv`
- `cusf_bessely_deriv`

## Helper mapping

CUSF helper-style names exposed in this compatibility module:

- `cusf_digamma`
- `cusf_gamma`
- `cusf_tgamma1pmv`
- `cusf_chebyshev`
- `cusf_polynomial`
- `cusf_poly_rational`

## Internal backend usage

The compatibility functions reuse existing JAX kernels in `arbPlusJAX` where possible:

- Hypergeometric and Bessel point kernels from `hypgeom.py`
- Interval mode dispatch from `baseline_wrappers.py` / `hypgeom_wrappers.py`
- Interval arithmetic from `double_interval.py`

This means the `cusf_*` layer is maintained as a naming/API compatibility surface, not a duplicate numerical backend.

## Notes

- `cusf_faddeeva_w` currently uses the project's complex `erfc` series path in pure JAX.
- For derivatives (`jvp`, `yvp`, `ivp`, `kvp`), point mode uses standard recurrence identities; interval modes apply the same identities over interval outputs.
