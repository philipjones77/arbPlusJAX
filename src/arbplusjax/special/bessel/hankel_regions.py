from __future__ import annotations

import jax.numpy as jnp


def choose_hankel_method(nu, z, *, requested_method: str = "auto") -> str:
    if requested_method != "auto":
        return requested_method

    nu_v = jnp.asarray(nu)
    z_v = jnp.asarray(z)
    if any(jnp.ndim(val) > 0 for val in (nu_v, z_v)):
        return "direct"

    z_abs = float(jnp.abs(z_v))
    if z_abs >= 16.0:
        return "asymptotic"
    return "direct"


__all__ = ["choose_hankel_method"]
