from __future__ import annotations

import jax.numpy as jnp


def choose_spherical_bessel_method(n, z, *, family: str, requested_method: str = "auto") -> str:
    if requested_method != "auto":
        return requested_method

    n_v = int(jnp.asarray(n, dtype=jnp.int32))
    z_v = jnp.asarray(z)
    if jnp.ndim(z_v) > 0:
        if family in {"j", "i"}:
            return "series"
        return "recurrence"

    z_abs = float(jnp.abs(z_v))
    if family in {"j", "i"} and z_abs < 0.75:
        return "series"
    if z_abs >= max(16.0, float(n_v + 8)):
        return "asymptotic"
    return "recurrence"


__all__ = ["choose_spherical_bessel_method"]
