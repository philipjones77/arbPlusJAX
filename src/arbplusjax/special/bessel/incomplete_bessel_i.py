from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ... import double_interval as di
from ..tail_acceleration import TailEvaluationDiagnostics
from ..tail_acceleration.quadrature import finite_interval_quadrature
from .incomplete_bessel_base import incomplete_bessel_i_angular_integrand
from .regions import choose_incomplete_bessel_i_method


def incomplete_bessel_i_point(
    nu,
    z,
    upper_limit,
    *,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
    return_diagnostics: bool = False,
):
    normalized_method = choose_incomplete_bessel_i_method(
        nu,
        z,
        upper_limit,
        requested_method=_normalize_method(method, nu, z, upper_limit),
    )
    if return_diagnostics:
        return _incomplete_bessel_i_point_base(
            nu,
            z,
            upper_limit,
            method=normalized_method,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
            return_diagnostics=True,
        )
    return _incomplete_bessel_i_point_ad(nu, z, upper_limit, normalized_method, panel_count, samples_per_panel)


def incomplete_bessel_i(
    nu,
    z,
    upper_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_count: int = 128,
    samples_per_panel: int = 16,
    return_diagnostics: bool = False,
):
    del dps
    point_value, diagnostics = incomplete_bessel_i_point(
        nu,
        z,
        upper_limit,
        method=method,
        panel_count=panel_count,
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
    if mode == "basic":
        radius = base_radius
    elif mode == "adaptive":
        radius = 1.5 * base_radius
    elif mode == "rigorous":
        radius = 3.0 * base_radius + remainder
    else:
        raise ValueError("mode must be one of: point, basic, adaptive, rigorous")

    interval = jnp.stack([point_value - radius, point_value + radius], axis=-1)
    interval = di.round_interval_outward(interval, pb)
    if return_diagnostics:
        return interval, diagnostics
    return interval


def _incomplete_bessel_i_point_base(
    nu,
    z,
    upper_limit,
    *,
    method: str,
    panel_count: int,
    samples_per_panel: int,
    return_diagnostics: bool,
):
    if method == "high_precision_refine":
        coarse_value, coarse_diag = _incomplete_bessel_i_quadrature(
            nu,
            z,
            upper_limit,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
            quadrature_rule="simpson",
            diagnostics_method="high_precision_refine",
            fallback_used=True,
            precision_warning=True,
        )
        refined_value, refined_diag = _incomplete_bessel_i_quadrature(
            nu,
            z,
            upper_limit,
            panel_count=max(panel_count * 2, 256),
            samples_per_panel=max(samples_per_panel * 2, 64),
            quadrature_rule="gauss_legendre",
            diagnostics_method="high_precision_refine",
            fallback_used=True,
            precision_warning=True,
        )
        consistency_err = jnp.abs(refined_value - coarse_value)
        estimated_remainder = jnp.maximum(
            jnp.asarray(refined_diag.estimated_tail_remainder, dtype=jnp.float64),
            jnp.asarray(consistency_err, dtype=jnp.float64),
        )
        instability_flags: list[str] = []
        scale = jnp.maximum(jnp.abs(refined_value), jnp.asarray(1e-12, dtype=jnp.float64))
        if bool(consistency_err > 0.1 * scale):
            instability_flags.append("fallback_self_consistency_warning")
        value = refined_value
        diagnostics = TailEvaluationDiagnostics(
            method="high_precision_refine",
            chunk_count=coarse_diag.chunk_count + refined_diag.chunk_count,
            panel_count=coarse_diag.panel_count + refined_diag.panel_count,
            recurrence_steps=0,
            estimated_tail_remainder=estimated_remainder,
            instability_flags=tuple(instability_flags),
            fallback_used=True,
            precision_warning=True,
            note="Pure-JAX refined angular quadrature fallback for fragile incomplete-I regimes.",
        )
    else:
        value, diagnostics = _incomplete_bessel_i_quadrature(
            nu,
            z,
            upper_limit,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
            quadrature_rule="simpson",
            diagnostics_method="quadrature",
            fallback_used=False,
            precision_warning=False,
        )
    if return_diagnostics:
        return value, diagnostics
    return value


def _normalize_method(method: str, *args) -> str:
    if method != "auto":
        return method
    if any(jnp.ndim(jnp.asarray(arg)) > 0 for arg in args):
        return "quadrature"
    return method


def _incomplete_bessel_i_quadrature(
    nu,
    z,
    upper_limit,
    *,
    panel_count: int,
    samples_per_panel: int,
    quadrature_rule: str,
    diagnostics_method: str,
    fallback_used: bool,
    precision_warning: bool,
):
    upper = jnp.asarray(upper_limit, dtype=jnp.float64)
    integrand = incomplete_bessel_i_angular_integrand(nu, z)
    value, diagnostics = finite_interval_quadrature(
        integrand,
        jnp.asarray(0.0, dtype=jnp.float64),
        upper,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
        quadrature_rule=quadrature_rule,
        method=diagnostics_method,
        fallback_used=fallback_used,
        precision_warning=precision_warning,
        note="Angular finite-interval quadrature for the incomplete-I truncation object.",
    )
    return value, diagnostics


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def _incomplete_bessel_i_point_ad(
    nu,
    z,
    upper_limit,
    method: str,
    panel_count: int,
    samples_per_panel: int,
):
    return _incomplete_bessel_i_point_base(
        nu,
        z,
        upper_limit,
        method=method,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )


@_incomplete_bessel_i_point_ad.defjvp
def _incomplete_bessel_i_point_ad_jvp(method, panel_count, samples_per_panel, primals, tangents):
    nu, z, upper_limit = primals
    t_nu, t_z, t_upper = tangents
    primal_out = _incomplete_bessel_i_point_base(
        nu,
        z,
        upper_limit,
        method=method,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )
    from .derivatives import incomplete_bessel_i_argument_derivative, incomplete_bessel_i_upper_limit_derivative

    step = jnp.asarray(1e-4, dtype=jnp.float64)
    nu_derivative = (
        _incomplete_bessel_i_point_base(
            nu + step,
            z,
            upper_limit,
            method=method,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
        - _incomplete_bessel_i_point_base(
            nu - step,
            z,
            upper_limit,
            method=method,
            panel_count=panel_count,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
    ) / (2.0 * step)
    z_derivative = incomplete_bessel_i_argument_derivative(
        nu,
        z,
        upper_limit,
        panel_count=panel_count,
        samples_per_panel=samples_per_panel,
    )
    upper_derivative = incomplete_bessel_i_upper_limit_derivative(nu, z, upper_limit)
    tangent_out = t_nu * nu_derivative + t_z * z_derivative + t_upper * upper_derivative
    return primal_out, tangent_out
