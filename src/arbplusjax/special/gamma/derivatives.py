from __future__ import annotations

from jax import lax
import jax.numpy as jnp

from .incomplete_gamma import incomplete_gamma_lower_integrand, incomplete_gamma_upper_integrand


def incomplete_gamma_upper_argument_derivative(s, z, *, regularized: bool = False):
    integrand = incomplete_gamma_upper_integrand(s)
    value = -integrand(z)
    if regularized:
        value = value / jnp.exp(lax.lgamma(jnp.asarray(s, dtype=jnp.float64)))
    return value


def incomplete_gamma_lower_argument_derivative(s, z, *, regularized: bool = False):
    integrand = incomplete_gamma_lower_integrand(s)
    value = integrand(z)
    if regularized:
        value = value / jnp.exp(lax.lgamma(jnp.asarray(s, dtype=jnp.float64)))
    return value


def incomplete_gamma_upper_parameter_derivative(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    from .incomplete_gamma import incomplete_gamma_upper_point

    step = jnp.asarray(1e-4, dtype=jnp.float64)
    return (
        incomplete_gamma_upper_point(
            jnp.asarray(s, dtype=jnp.float64) + step,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
        - incomplete_gamma_upper_point(
            jnp.asarray(s, dtype=jnp.float64) - step,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    ) / (2.0 * step)


def incomplete_gamma_lower_parameter_derivative(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    from .incomplete_gamma import incomplete_gamma_lower_point

    step = jnp.asarray(1e-4, dtype=jnp.float64)
    return (
        incomplete_gamma_lower_point(
            jnp.asarray(s, dtype=jnp.float64) + step,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
        - incomplete_gamma_lower_point(
            jnp.asarray(s, dtype=jnp.float64) - step,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    ) / (2.0 * step)


def incomplete_gamma_upper_derivative(
    s,
    z,
    *,
    respect_to: str = "z",
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    if respect_to == "z":
        return incomplete_gamma_upper_argument_derivative(s, z, regularized=regularized)
    if respect_to == "s":
        return incomplete_gamma_upper_parameter_derivative(
            s,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    raise ValueError("respect_to must be one of: z, s")


def incomplete_gamma_lower_derivative(
    s,
    z,
    *,
    respect_to: str = "z",
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    if respect_to == "z":
        return incomplete_gamma_lower_argument_derivative(s, z, regularized=regularized)
    if respect_to == "s":
        return incomplete_gamma_lower_parameter_derivative(
            s,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    raise ValueError("respect_to must be one of: z, s")
