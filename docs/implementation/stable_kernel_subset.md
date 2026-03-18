Last updated: 2026-03-17T00:00:00Z

# Stable Kernel Subset

This repo now exposes a curated downstream-facing kernel subset in `src/arbplusjax/stable_kernels.py`.

The intent is interface stability and routing clarity, not a claim that every member family has identical hardening maturity.

## Curated kernels

- `gamma`
- `loggamma` (downstream alias of the canonical `lgamma` surface)
- `incomplete_gamma_lower`
- `incomplete_gamma_upper`
- `incomplete_bessel_i`
- `incomplete_bessel_k`

## Routing source of truth

- Machine-readable capability registry: `docs/reports/function_capability_registry.json`
- Runtime builder: `src/arbplusjax/capability_registry.py`
- Downstream wrappers: `src/arbplusjax/stable_kernels.py`

## Notes

- `gamma` and `loggamma` route to the canonical public API and keep the standard `point|basic|adaptive|rigorous` mode contract.
- The incomplete gamma and incomplete Bessel kernels remain tagged as experimental in the broader public metadata, but they are the supported downstream entry points for those families in this repo.
- Downstream code should prefer the stable-kernel aliases over implementation-lineage names when the goal is routing by capability rather than by provenance.
