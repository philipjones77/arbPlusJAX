Last updated: 2026-03-18T00:00:00Z

# special.bessel

## Scope

- Higher-level Bessel-family surfaces that sit above the low-level `hypgeom.py` and `bessel_kernels.py` kernels.
- Complete Hankel functions `H_nu^(1)` and `H_nu^(2)`.
- Integer-order spherical Bessel families `j_n`, `y_n`, `i_n`, `k_n`.
- Incomplete Bessel tails remain in the same package because they share recurrence/asymptotic/fallback organization.

## Code layout

- `src/arbplusjax/special/bessel/hankel.py`
- `src/arbplusjax/special/bessel/hankel_asymptotics.py`
- `src/arbplusjax/special/bessel/hankel_recurrences.py`
- `src/arbplusjax/special/bessel/hankel_derivatives.py`
- `src/arbplusjax/special/bessel/hankel_regions.py`
- `src/arbplusjax/special/bessel/spherical.py`
- `src/arbplusjax/special/bessel/spherical_asymptotics.py`
- `src/arbplusjax/special/bessel/spherical_recurrences.py`
- `src/arbplusjax/special/bessel/spherical_derivatives.py`
- `src/arbplusjax/special/bessel/spherical_regions.py`

## Public surfaces

- `arbplusjax.special.bessel.hankel1`
- `arbplusjax.special.bessel.hankel2`
- `arbplusjax.special.bessel.scaled_hankel1`
- `arbplusjax.special.bessel.scaled_hankel2`
- `arbplusjax.special.bessel.spherical_bessel_j`
- `arbplusjax.special.bessel.spherical_bessel_y`
- `arbplusjax.special.bessel.modified_spherical_bessel_i`
- `arbplusjax.special.bessel.modified_spherical_bessel_k`

The same point-evaluation surfaces are also registered in `arbplusjax.api` under:

- `hankel1`
- `hankel2`
- `scaled_hankel1`
- `scaled_hankel2`
- `spherical_bessel_j`
- `spherical_bessel_y`
- `modified_spherical_bessel_i`
- `modified_spherical_bessel_k`

## Implementation notes

### Hankel

- Direct path: `H_nu^(1) = J_nu + i Y_nu`, `H_nu^(2) = J_nu - i Y_nu`.
- Large-`|z|` path: leading cylindrical Hankel asymptotic.
- Derivatives: adjacent-order identity `dH_nu/dz = 0.5 (H_{nu-1} - H_{nu+1})`.
- Scaling:
  - `scaled_hankel1 = exp(-i z) H_nu^(1)(z)`
  - `scaled_hankel2 = exp(+i z) H_nu^(2)(z)`

### Spherical

- The implementation is concrete and family-specific rather than a public alias to half-integer cylindrical Bessel.
- `j_n` and `i_n` use:
  - a small-`|z|` series path,
  - an integer-order upward recurrence path,
  - a large-`|z|` asymptotic path.
- `y_n` and `k_n` use:
  - explicit seed formulas at orders `0` and `1`,
  - upward recurrences,
  - a large-`|z|` asymptotic path.
- Derivatives use the standard integer-order spherical identities.

## Current status

- These surfaces are point-evaluation functions.
- They are batchable through `api.eval_point_batch(...)`.
- They are not yet wired into the interval/ball wrapper layer as dedicated rigorous/adaptive enclosures.
- The module placement is intended to keep that future work local to the Bessel package instead of scattering it across application libraries.
