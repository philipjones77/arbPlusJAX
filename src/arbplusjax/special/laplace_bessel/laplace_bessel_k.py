from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ... import double_interval as di
from ... import point_wrappers
from ..tail_acceleration import (
    TailDerivativeMetadata,
    TailEvaluationDiagnostics,
    TailIntegralProblem,
    evaluate_tail_integral,
)
from .regions import choose_laplace_bessel_k_tail_method, laplace_bessel_k_tail_regime_metadata


def laplace_bessel_k_tail_integrand(nu, z, lam):
    nu_val = jnp.asarray(nu, dtype=jnp.float64)
    z_val = jnp.asarray(z, dtype=jnp.float64)
    lam_val = jnp.asarray(lam, dtype=jnp.float64)

    def integrand(t):
        t_val = jnp.asarray(t, dtype=jnp.float64)
        return jnp.exp(-lam_val * t_val) * point_wrappers.arb_bessel_k_point(nu_val, z_val * t_val)

    return integrand


def build_laplace_bessel_k_tail_problem(
    nu,
    z,
    lam,
    lower_limit,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
) -> TailIntegralProblem:
    return TailIntegralProblem(
        integrand=laplace_bessel_k_tail_integrand(nu, z, lam),
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        quadrature_rule="simpson",
        derivative_metadata=TailDerivativeMetadata(
            argument_derivative=False,
            lower_limit_derivative=True,
            parameter_derivative=True,
            note="Explicit lower-limit and Laplace-parameter derivatives are exposed for Laplace-Bessel-K tails.",
        ),
        regime_metadata=laplace_bessel_k_tail_regime_metadata(nu, z, lam, lower_limit),
        name="laplace_bessel_k_tail",
    )


def laplace_bessel_k_tail_point(
    nu,
    z,
    lam,
    lower_limit,
    *,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
    return_diagnostics: bool = False,
):
    normalized_method = choose_laplace_bessel_k_tail_method(
        nu,
        z,
        lam,
        lower_limit,
        requested_method=_normalize_method(method, nu, z, lam, lower_limit),
    )
    if return_diagnostics:
        return _laplace_bessel_k_tail_point_base(
            nu,
            z,
            lam,
            lower_limit,
            method=normalized_method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=True,
        )
    return _laplace_bessel_k_tail_point_ad(
        nu,
        z,
        lam,
        lower_limit,
        normalized_method,
        panel_width,
        max_panels,
        samples_per_panel,
    )


def laplace_bessel_k_tail(
    nu,
    z,
    lam,
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
    point_value, diagnostics = laplace_bessel_k_tail_point(
        nu,
        z,
        lam,
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


def laplace_bessel_k_tail_batch(
    nu,
    z,
    lam,
    lower_limit,
    *,
    mode: str = "point",
    prec_bits: int | None = None,
    dps: int | None = None,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    fn = lambda a, b, c, d: laplace_bessel_k_tail(
        a,
        b,
        c,
        d,
        mode=mode,
        prec_bits=prec_bits,
        dps=dps,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    return jax.vmap(fn)(nu, z, lam, lower_limit)


def laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower_limit):
    integrand = laplace_bessel_k_tail_integrand(nu, z, lam)
    return -integrand(lower_limit)


def laplace_bessel_k_tail_lambda_derivative(
    nu,
    z,
    lam,
    lower_limit,
    *,
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    nu_val = jnp.asarray(nu, dtype=jnp.float64)
    z_val = jnp.asarray(z, dtype=jnp.float64)
    lam_val = jnp.asarray(lam, dtype=jnp.float64)

    def weighted_integrand(t):
        t_val = jnp.asarray(t, dtype=jnp.float64)
        return -t_val * jnp.exp(-lam_val * t_val) * point_wrappers.arb_bessel_k_point(nu_val, z_val * t_val)

    problem = TailIntegralProblem(
        integrand=weighted_integrand,
        lower_limit=lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        derivative_metadata=TailDerivativeMetadata(lower_limit_derivative=False, parameter_derivative=False),
        regime_metadata=laplace_bessel_k_tail_regime_metadata(nu, z, lam, lower_limit),
        name="laplace_bessel_k_tail_lambda_derivative",
    )
    return evaluate_tail_integral(problem, method=method, return_diagnostics=False)


def laplace_bessel_k_tail_derivative(
    nu,
    z,
    lam,
    lower_limit,
    *,
    respect_to: str = "lambda",
    method: str = "quadrature",
    panel_width: float = 0.125,
    max_panels: int = 160,
    samples_per_panel: int = 24,
):
    if respect_to == "lambda":
        return laplace_bessel_k_tail_lambda_derivative(
            nu,
            z,
            lam,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
        )
    if respect_to == "lower_limit":
        return laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower_limit)
    raise ValueError("respect_to must be one of: lambda, lower_limit")


def _normalize_method(method: str, *args) -> str:
    if method != "auto":
        return method
    if any(jnp.ndim(jnp.asarray(arg)) > 0 for arg in args):
        return "quadrature"
    return method


def _laplace_bessel_k_tail_point_base(
    nu,
    z,
    lam,
    lower_limit,
    *,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    return_diagnostics: bool,
):
    problem = build_laplace_bessel_k_tail_problem(
        nu,
        z,
        lam,
        lower_limit,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    value, diagnostics = evaluate_tail_integral(problem, method=method, return_diagnostics=True)
    if return_diagnostics:
        return value, diagnostics
    return value


@partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7))
def _laplace_bessel_k_tail_point_ad(
    nu,
    z,
    lam,
    lower_limit,
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
):
    return _laplace_bessel_k_tail_point_base(
        nu,
        z,
        lam,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )


@_laplace_bessel_k_tail_point_ad.defjvp
def _laplace_bessel_k_tail_point_ad_jvp(
    method: str,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    primals,
    tangents,
):
    nu, z, lam, lower_limit = primals
    t_nu, t_z, t_lam, t_lower = tangents
    primal_out = _laplace_bessel_k_tail_point_base(
        nu,
        z,
        lam,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
        return_diagnostics=False,
    )
    step = jnp.asarray(1e-4, dtype=jnp.float64)
    nu_derivative = (
        _laplace_bessel_k_tail_point_base(
            nu + step,
            z,
            lam,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
        - _laplace_bessel_k_tail_point_base(
            nu - step,
            z,
            lam,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
    ) / (2.0 * step)
    z_derivative = (
        _laplace_bessel_k_tail_point_base(
            nu,
            z + step,
            lam,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
        - _laplace_bessel_k_tail_point_base(
            nu,
            z - step,
            lam,
            lower_limit,
            method=method,
            panel_width=panel_width,
            max_panels=max_panels,
            samples_per_panel=samples_per_panel,
            return_diagnostics=False,
        )
    ) / (2.0 * step)
    lam_derivative = laplace_bessel_k_tail_lambda_derivative(
        nu,
        z,
        lam,
        lower_limit,
        method=method,
        panel_width=panel_width,
        max_panels=max_panels,
        samples_per_panel=samples_per_panel,
    )
    lower_derivative = laplace_bessel_k_tail_lower_limit_derivative(nu, z, lam, lower_limit)
    tangent_out = t_nu * nu_derivative + t_z * z_derivative + t_lam * lam_derivative + t_lower * lower_derivative
    return primal_out, tangent_out
