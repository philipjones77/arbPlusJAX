from __future__ import annotations

import jax
import jax.numpy as jnp

from .incomplete_bessel_base import incomplete_bessel_i_angular_integrand, incomplete_bessel_k_integrand


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


def incomplete_bessel_i_upper_limit_derivative(
    nu,
    z,
    upper_limit,
    *,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
):
    del method, panel_count, samples_per_panel
    integrand = incomplete_bessel_i_angular_integrand(nu, z)
    return integrand(upper_limit)


def incomplete_bessel_i_argument_derivative(
    nu,
    z,
    upper_limit,
    *,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
):
    del method
    upper = jnp.asarray(upper_limit, dtype=jnp.float64)
    if panel_count <= 0:
        raise ValueError("panel_count must be > 0")
    width = upper / jnp.asarray(panel_count, dtype=jnp.float64)
    offsets = jnp.linspace(0.0, 1.0, samples_per_panel, dtype=jnp.float64)
    indices = jnp.arange(panel_count, dtype=jnp.int32)
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    z_v = jnp.asarray(z, dtype=jnp.float64)

    def integrand(theta):
        tv = jnp.asarray(theta, dtype=jnp.float64)
        return jnp.cos(tv) * jnp.exp(z_v * jnp.cos(tv)) * jnp.cos(nu_v * tv)

    def panel_step(total, panel_index):
        start = jnp.asarray(panel_index, dtype=jnp.float64) * width
        grid = start + width * offsets
        vals = jax.vmap(integrand)(grid)
        return total + jnp.trapezoid(vals, grid), None

    total, _ = jax.lax.scan(panel_step, jnp.asarray(0.0, dtype=jnp.float64), indices)
    return total


def incomplete_bessel_i_derivative(
    nu,
    z,
    upper_limit,
    *,
    respect_to: str = "z",
    panel_count: int = 128,
    samples_per_panel: int = 16,
):
    if respect_to == "z":
        return incomplete_bessel_i_argument_derivative(
            nu,
            z,
            upper_limit,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
        )
    if respect_to == "upper_limit":
        return incomplete_bessel_i_upper_limit_derivative(
            nu,
            z,
            upper_limit,
            method=method,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
        )
    raise ValueError("respect_to must be one of: z, upper_limit")
