from __future__ import annotations

import jax.numpy as jnp

from .incomplete_bessel_base import incomplete_bessel_k_integrand


def incomplete_bessel_k_lower_limit_derivative(nu, z, lower_limit):
    integrand = incomplete_bessel_k_integrand(nu, z)
    return -integrand(lower_limit)


def incomplete_bessel_k_argument_derivative(
    nu,
    z,
    lower_limit,
    *,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    from .incomplete_bessel_k import incomplete_bessel_k_point

    upper = incomplete_bessel_k_point(
        jnp.asarray(nu, dtype=jnp.float64) + jnp.asarray(1.0, dtype=jnp.float64),
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    lower = incomplete_bessel_k_point(
        jnp.asarray(nu, dtype=jnp.float64) - jnp.asarray(1.0, dtype=jnp.float64),
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return -0.5 * (upper + lower)


def incomplete_bessel_k_derivative(
    nu,
    z,
    lower_limit,
    *,
    respect_to: str = "z",
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    if respect_to == "z":
        return incomplete_bessel_k_argument_derivative(
            nu,
            z,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    if respect_to == "lower_limit":
        return incomplete_bessel_k_lower_limit_derivative(nu, z, lower_limit)
    raise ValueError("respect_to must be one of: z, lower_limit")
