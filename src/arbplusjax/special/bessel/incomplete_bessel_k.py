from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ... import double_interval as di
from ..tail_acceleration import TailEvaluationDiagnostics, evaluate_tail_integral
from .asymptotics import incomplete_bessel_k_asymptotic
from .fallback_mp import incomplete_bessel_k_mpfallback
from .incomplete_bessel_base import build_incomplete_bessel_k_problem
from .recurrences import incomplete_bessel_k_recurrence
from .regions import choose_incomplete_bessel_k_method


def incomplete_bessel_k_point(
    nu,
    z,
    lower_limit,
    *,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, TailEvaluationDiagnostics]:
    normalized_method = choose_incomplete_bessel_k_method(
        nu,
        z,
        lower_limit,
        requested_method=_normalize_method(method, nu, z, lower_limit),
    )
    if return_diagnostics:
        return _incomplete_bessel_k_point_base(
            nu,
            z,
            lower_limit,
            method=normalized_method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=True,
        )
    return _incomplete_bessel_k_point_ad(
        nu,
        z,
        lower_limit,
        normalized_method,
        panel_width,
        max_panels,
        samples_per_panel,
    )


def incomplete_bessel_k(
    nu,
    z,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    del dps
    point_value, diagnostics = incomplete_bessel_k_point(
        nu,
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=True,
    )
    if mode == "point":
        if return_diagnostics:
            return point_value, diagnostics
        return point_value

    point_value = jnp.asarray(point_value, dtype=jnp.float64)
    pb = 53 if prec_bits is None else int(prec_bits)
    eps = jnp.exp2(jnp.asarray(-pb, dtype=jnp.float64))
    remainder = jnp.asarray(diagnostics.estimated_tail_remainder, dtype=jnp.float64)
    base_radius = jnp.abs(point_value) * eps + remainder + eps
    adaptive_scale = jnp.asarray(2.0 if diagnostics.method in {"aitken", "wynn"} else 1.5, dtype=jnp.float64)
    rigorous_scale = jnp.asarray(4.0 if diagnostics.fallback_used else 3.0, dtype=jnp.float64)
    if mode == "basic":
        radius = base_radius
    elif mode == "adaptive":
        radius = adaptive_scale * base_radius
    elif mode == "rigorous":
        radius = rigorous_scale * base_radius + remainder
    else:
        raise ValueError("mode must be one of: point, basic, adaptive, rigorous")

    interval = jnp.stack([point_value - radius, point_value + radius], axis=-1)
    interval = di.round_interval_outward(interval, pb)
    if return_diagnostics:
        return interval, diagnostics
    return interval


def _normalize_method(method: str, *args) -> str:
    if method != "auto":
        return method
    if any(jnp.ndim(jnp.asarray(arg)) > 0 for arg in args):
        return "quadrature"
    return method


def _incomplete_bessel_k_point_base(
    nu,
    z,
    lower_limit,
    *,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    return_diagnostics: bool,
):
    if method == "asymptotic":
        value, diagnostics = incomplete_bessel_k_asymptotic(nu, z, lower_limit)
    elif method == "recurrence":
        value, diagnostics = incomplete_bessel_k_recurrence(nu, z, lower_limit)
    elif method == "high_precision_refine":
        value, diagnostics = incomplete_bessel_k_mpfallback(
            nu,
            z,
            lower_limit,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    else:
        problem = build_incomplete_bessel_k_problem(
            nu,
            z,
            lower_limit,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
        value, diagnostics = evaluate_tail_integral(problem, method=method, return_diagnostics=True)
    if return_diagnostics:
        return value, diagnostics
    return value


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6))
def _incomplete_bessel_k_point_ad(
    nu,
    z,
    lower_limit,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
):
    return _incomplete_bessel_k_point_base(
        nu,
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )


@_incomplete_bessel_k_point_ad.defjvp
def _incomplete_bessel_k_point_ad_jvp(
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    primals,
    tangents,
):
    nu, z, lower_limit = primals
    t_nu, t_z, t_lower = tangents
    primal_out = _incomplete_bessel_k_point_base(
        nu,
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )
    from .derivatives import incomplete_bessel_k_argument_derivative, incomplete_bessel_k_lower_limit_derivative

    step = jnp.asarray(1e-4, dtype=jnp.float64)
    nu_derivative = (
        _incomplete_bessel_k_point_base(
            nu + step,
            z,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
        - _incomplete_bessel_k_point_base(
            nu - step,
            z,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
    ) / (2.0 * step)
    z_derivative = incomplete_bessel_k_argument_derivative(
        nu,
        z,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    lower_derivative = incomplete_bessel_k_lower_limit_derivative(nu, z, lower_limit)
    tangent_out = t_nu * nu_derivative + t_z * z_derivative + t_lower * lower_derivative
    return primal_out, tangent_out
