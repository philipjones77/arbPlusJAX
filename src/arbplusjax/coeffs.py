from __future__ import annotations

import jax
import jax.numpy as jnp


# Shared Lanczos table used by log-gamma implementations.
LANCZOS = jnp.asarray(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ],
    dtype=jnp.float64,
)

# Shared Stirling-series coefficients.
STIRLING_COEFFS = jnp.asarray(
    [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
        1.0 / 156.0,
        -3617.0 / 122400.0,
    ],
    dtype=jnp.float64,
)


__all__ = ["LANCZOS", "STIRLING_COEFFS"]
