from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ... import double_interval as di
from ..tail_acceleration import (
    TailDerivativeMetadata,
    TailEvaluationDiagnostics,
    TailIntegralProblem,
    evaluate_tail_integral,
)
from .regions import choose_incomplete_gamma_upper_method, incomplete_gamma_upper_regime_metadata


def incomplete_gamma_upper_integrand(s):
    s_val = jnp.asarray(s, dtype=jnp.float64)

    def integrand(t: jax.Array) -> jax.Array:
        t_val = jnp.asarray(t, dtype=jnp.float64)
        safe_t = jnp.maximum(t_val, jnp.asarray(1e-300, dtype=jnp.float64))
        return jnp.exp((s_val - 1.0) * jnp.log(safe_t) - t_val)

    return integrand


def incomplete_gamma_lower_integrand(s):
    return incomplete_gamma_upper_integrand(s)


def build_incomplete_gamma_upper_problem(
    s,
    z,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
) -> TailIntegralProblem:
    return TailIntegralProblem(
        integrand=incomplete_gamma_upper_integrand(s),
        lower_limit=z,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        quadrature_rule="simpson",
        derivative_metadata=TailDerivativeMetadata(
            argument_derivative=True,
            lower_limit_derivative=True,
            parameter_derivative=True,
            note="Explicit z-derivative and AD-facing custom JVP support are available for incomplete-gamma upper tails.",
        ),
        regime_metadata=incomplete_gamma_upper_regime_metadata(s, z),
        name="incomplete_gamma_upper",
    )


def incomplete_gamma_upper_point(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    normalized_method = choose_incomplete_gamma_upper_method(
        s,
        z,
        requested_method=_normalize_method(method, s, z),
    )
    if return_diagnostics:
        return _incomplete_gamma_upper_point_base(
            s,
            z,
            regularized=regularized,
            method=normalized_method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=True,
        )
    return _incomplete_gamma_upper_point_ad(
        s,
        z,
        regularized,
        normalized_method,
        panel_width,
        max_panels,
        samples_per_panel,
    )


def incomplete_gamma_upper(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    del dps
    point_value, diagnostics = incomplete_gamma_upper_point(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=True,
    )
    return _wrap_mode_output(
        point_value,
        diagnostics,
        mode=mode,
        prec_bits=prec_bits,
        return_diagnostics=return_diagnostics,
    )


def incomplete_gamma_upper_batch(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b: incomplete_gamma_upper(
        a,
        b,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(s, z)


def incomplete_gamma_lower_point(
    s,
    z,
    *,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    if return_diagnostics:
        return _incomplete_gamma_lower_point_base(
            s,
            z,
            regularized=regularized,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=True,
        )
    normalized_method = choose_incomplete_gamma_upper_method(
        s,
        z,
        requested_method=_normalize_method(method, s, z),
    )
    return _incomplete_gamma_lower_point_ad(
        s,
        z,
        regularized,
        normalized_method,
        panel_width,
        max_panels,
        samples_per_panel,
    )


def incomplete_gamma_lower(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    del dps
    point_value, diagnostics = incomplete_gamma_lower_point(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=True,
    )
    return _wrap_mode_output(
        point_value,
        diagnostics,
        mode=mode,
        prec_bits=prec_bits,
        return_diagnostics=return_diagnostics,
    )


def incomplete_gamma_lower_batch(
    s,
    z,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    regularized: bool = False,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b: incomplete_gamma_lower(
        a,
        b,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(s, z)


def _normalize_method(method: str, *args) -> str:
    if method != "auto":
        return method
    if any(jnp.ndim(jnp.asarray(arg)) > 0 for arg in args):
        return "quadrature"
    return method


def _incomplete_gamma_upper_point_base(
    s,
    z,
    *,
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    return_diagnostics: bool,
):
    problem = build_incomplete_gamma_upper_problem(
        s,
        z,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    value, diagnostics = evaluate_tail_integral(problem, method=method, return_diagnostics=True)
    if regularized:
        value = value / jnp.exp(lax.lgamma(jnp.asarray(s, dtype=jnp.float64)))
    if return_diagnostics:
        return value, diagnostics
    return value


def _incomplete_gamma_lower_point_base(
    s,
    z,
    *,
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    return_diagnostics: bool,
):
    upper_value, upper_diagnostics = _incomplete_gamma_upper_point_base(
        s,
        z,
        regularized=False,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=True,
    )
    gamma_s = jnp.exp(lax.lgamma(jnp.asarray(s, dtype=jnp.float64)))
    value = gamma_s - upper_value
    if regularized:
        value = value / gamma_s
    diagnostics = TailEvaluationDiagnostics(
        method=upper_diagnostics.method,
        chunk_count=upper_diagnostics.chunk_count,
        panel_count=upper_diagnostics.panel_count,
        recurrence_steps=upper_diagnostics.recurrence_steps,
        estimated_tail_remainder=upper_diagnostics.estimated_tail_remainder,
        instability_flags=upper_diagnostics.instability_flags,
        fallback_used=upper_diagnostics.fallback_used,
        precision_warning=upper_diagnostics.precision_warning,
        note="Complement-backed incomplete-gamma lower evaluation built from the upper tail integral specialization.",
    )
    if return_diagnostics:
        return value, diagnostics
    return value


@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4, 5, 6))
def _incomplete_gamma_upper_point_ad(
    s,
    z,
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
):
    return _incomplete_gamma_upper_point_base(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )


@_incomplete_gamma_upper_point_ad.defjvp
def _incomplete_gamma_upper_point_ad_jvp(
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    primals,
    tangents,
):
    s, z = primals
    t_s, t_z = tangents
    primal_out = _incomplete_gamma_upper_point_base(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )
    from .derivatives import incomplete_gamma_upper_argument_derivative, incomplete_gamma_upper_parameter_derivative

    s_derivative = incomplete_gamma_upper_parameter_derivative(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    z_derivative = incomplete_gamma_upper_argument_derivative(s, z, regularized=regularized)
    tangent_out = t_s * s_derivative + t_z * z_derivative
    return primal_out, tangent_out


@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4, 5, 6))
def _incomplete_gamma_lower_point_ad(
    s,
    z,
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
):
    return _incomplete_gamma_lower_point_base(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )


@_incomplete_gamma_lower_point_ad.defjvp
def _incomplete_gamma_lower_point_ad_jvp(
    regularized: bool,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    primals,
    tangents,
):
    s, z = primals
    t_s, t_z = tangents
    primal_out = _incomplete_gamma_lower_point_base(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )
    from .derivatives import incomplete_gamma_lower_argument_derivative, incomplete_gamma_lower_parameter_derivative

    s_derivative = incomplete_gamma_lower_parameter_derivative(
        s,
        z,
        regularized=regularized,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    z_derivative = incomplete_gamma_lower_argument_derivative(s, z, regularized=regularized)
    tangent_out = t_s * s_derivative + t_z * z_derivative
    return primal_out, tangent_out


def _wrap_mode_output(
    point_value,
    diagnostics: TailEvaluationDiagnostics,
    *,
    mode: str,
    prec_bits: int | None,
    return_diagnostics: bool,
):
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
