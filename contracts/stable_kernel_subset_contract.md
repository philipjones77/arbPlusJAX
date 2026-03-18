Last updated: 2026-03-17T00:00:00Z

# Stable Kernel Subset Contract

## Scope

This contract covers the downstream-facing kernel subset exported by `arbplusjax.stable_kernels`.

## Stable downstream aliases

The supported stable-kernel names are:

- `gamma`
- `loggamma`
- `incomplete_gamma_lower`
- `incomplete_gamma_upper`
- `incomplete_bessel_i`
- `incomplete_bessel_k`

The corresponding batch helpers are part of the same stable subset:

- `gamma_batch`
- `loggamma_batch`
- `incomplete_gamma_lower_batch`
- `incomplete_gamma_upper_batch`
- `incomplete_bessel_i_batch`
- `incomplete_bessel_k_batch`

## Routing contract

- `stable_kernels.list_supported_kernels()` must list the alias set above.
- `stable_kernels.get_kernel_capability(name)` must return the capability-registry entry for a supported alias.
- `gamma` and `loggamma` are downstream-facing aliases over the canonical public gamma surfaces.
- The incomplete gamma and incomplete Bessel entries are the supported downstream entry points for those families even though the broader metadata still classifies those families as experimental.

## Mode and dtype contract

- The stable-kernel wrappers use the standard public mode set: `point`, `basic`, `adaptive`, `rigorous`.
- Invalid mode names must raise an error rather than silently changing behavior.
- Optional `dtype=...` arguments act as input casting controls and do not redefine the repo-wide precision model.

## Compatibility contract

- Downstream code should route by stable-kernel alias when it wants capability-based access rather than implementation-lineage names.
- The exact underlying implementation may change, but the stable alias names and their routing role should remain stable unless a replacement contract is published.

## Source of truth

- `src/arbplusjax/stable_kernels.py`
- `src/arbplusjax/capability_registry.py`
- `docs/implementation/stable_kernel_subset.md`
- `tests/test_stable_kernels.py`

