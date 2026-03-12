from __future__ import annotations

import jax
import jax.numpy as jnp

from ..tail_acceleration import TailDerivativeMetadata, TailIntegralProblem
from .regions import incomplete_bessel_k_regime_metadata


def incomplete_bessel_k_integrand(nu, z):
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)

    def integrand(t: jax.Array) -> jax.Array:
        tv = jnp.asarray(t, dtype=jnp.float64)
        return jnp.exp(-z_v * jnp.cosh(tv)) * jnp.cosh(nu_v * tv)

    return integrand


def build_incomplete_bessel_k_problem(
    nu,
    z,
    lower_limit,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
) -> TailIntegralProblem:
    return TailIntegralProblem(
        integrand=incomplete_bessel_k_integrand(nu, z),
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        derivative_metadata=TailDerivativeMetadata(
            argument_derivative=True,
            lower_limit_derivative=True,
            parameter_derivative=False,
            note="Explicit z and lower-limit derivative identities are exposed for incomplete-K.",
        ),
        regime_metadata=incomplete_bessel_k_regime_metadata(nu, z, lower_limit),
        name="incomplete_bessel_k",
    )
