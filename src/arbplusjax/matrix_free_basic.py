from __future__ import annotations

import jax
import jax.numpy as jnp

from . import matrix_free_core


def operator_apply_basic(point_apply_fn, operator, x: jax.Array, *, round_output, prec_bits: int):
    return round_output(point_apply_fn(operator, x), prec_bits)


def action_basic(point_action_fn, operator, x: jax.Array, *args, round_output, prec_bits: int, **kwargs):
    return round_output(point_action_fn(operator, x, *args, **kwargs), prec_bits)


def action_with_diagnostics_basic(
    point_action_with_diagnostics_fn,
    operator,
    x: jax.Array,
    *args,
    round_output,
    prec_bits: int,
    inflate_output=None,
    invalidate_output=None,
    **kwargs,
):
    value, diagnostics = point_action_with_diagnostics_fn(operator, x, *args, **kwargs)
    value = round_output(value, prec_bits)
    if inflate_output is not None:
        value = inflate_output(value, diagnostics)
    value = _invalidate_action_if_unsafe(value, diagnostics, invalidate_output=invalidate_output)
    return value, diagnostics


def solve_action_basic(point_solve_fn, operator, b: jax.Array, *, round_output, prec_bits: int, **kwargs):
    return round_output(point_solve_fn(operator, b, **kwargs), prec_bits)


def solve_action_with_diagnostics_basic(
    point_solve_with_diagnostics_fn,
    operator,
    b: jax.Array,
    *,
    round_output,
    prec_bits: int,
    invalidate_output=None,
    **kwargs,
):
    value, diagnostics = point_solve_with_diagnostics_fn(operator, b, **kwargs)
    value = round_output(value, prec_bits)
    value = _invalidate_if_residual_unsafe(
        value,
        diagnostics,
        tol=kwargs.get("tol", 1e-8),
        atol=kwargs.get("atol", 0.0),
        invalidate_output=invalidate_output,
    )
    return value, diagnostics


def inverse_action_basic(point_inverse_fn, operator, x: jax.Array, *, round_output, prec_bits: int, **kwargs):
    return round_output(point_inverse_fn(operator, x, **kwargs), prec_bits)


def inverse_action_with_diagnostics_basic(
    point_inverse_with_diagnostics_fn,
    operator,
    x: jax.Array,
    *,
    round_output,
    prec_bits: int,
    invalidate_output=None,
    **kwargs,
):
    value, diagnostics = point_inverse_with_diagnostics_fn(operator, x, **kwargs)
    value = round_output(value, prec_bits)
    value = _invalidate_if_residual_unsafe(
        value,
        diagnostics,
        tol=kwargs.get("tol", 1e-8),
        atol=kwargs.get("atol", 0.0),
        invalidate_output=invalidate_output,
    )
    return value, diagnostics


def scalar_functional_basic(point_scalar_fn, *args, lift_scalar, round_output, prec_bits: int, **kwargs):
    return round_output(lift_scalar(point_scalar_fn(*args, **kwargs)), prec_bits)


def vector_functional_basic(point_vector_fn, *args, round_output, prec_bits: int, **kwargs):
    return round_output(point_vector_fn(*args, **kwargs), prec_bits)


def scalar_functional_with_diagnostics_basic(
    point_scalar_with_diagnostics_fn,
    *args,
    lift_scalar,
    round_output,
    prec_bits: int,
    inflate_output=None,
    invalidate_output=None,
    **kwargs,
):
    value, diagnostics = point_scalar_with_diagnostics_fn(*args, **kwargs)
    value = round_output(lift_scalar(value), prec_bits)
    if inflate_output is not None:
        value = inflate_output(value, diagnostics)
    value = _invalidate_scalar_if_unsafe(value, diagnostics, invalidate_output=invalidate_output)
    return value, diagnostics


def det_basic_from_logdet(point_logdet_fn, *args, lift_scalar, round_output, prec_bits: int, **kwargs):
    logdet = point_logdet_fn(*args, **kwargs)
    return round_output(lift_scalar(matrix_free_core.det_from_logdet(logdet)), prec_bits)


def scalar_uncertainty_radius(diagnostics):
    tail_norm = jnp.asarray(getattr(diagnostics, "tail_norm", 0.0), dtype=jnp.float64)
    primal_residual = jnp.asarray(getattr(diagnostics, "primal_residual", 0.0), dtype=jnp.float64)
    metric = jnp.asarray(getattr(diagnostics, "convergence_metric", tail_norm), dtype=jnp.float64)
    stderr = jnp.asarray(getattr(diagnostics, "stderr", 0.0), dtype=jnp.float64)
    radius = jnp.maximum(
        jnp.maximum(jnp.maximum(jnp.abs(tail_norm), jnp.abs(primal_residual)), jnp.abs(metric)),
        jnp.abs(stderr),
    )
    return jnp.where(jnp.isfinite(radius), radius, 0.0)


def _residual_limit(diag, *, tol: float, atol: float):
    beta0 = jnp.asarray(getattr(diag, "beta0", 1.0), dtype=jnp.float64)
    scale = jnp.maximum(jnp.abs(beta0), 1.0)
    return jnp.maximum(jnp.asarray(atol, dtype=jnp.float64), jnp.asarray(tol, dtype=jnp.float64) * scale)


def _invalidate_if_residual_unsafe(value, diagnostics, *, tol: float, atol: float, invalidate_output):
    if invalidate_output is None:
        return value
    residual = jnp.asarray(getattr(diagnostics, "primal_residual", jnp.inf), dtype=jnp.float64)
    limit = _residual_limit(diagnostics, tol=tol, atol=atol)
    ok = jnp.isfinite(residual) & (residual <= limit)
    while ok.ndim < value.ndim:
        ok = ok[..., None]
    return jnp.where(ok, value, invalidate_output(value))


def _invalidate_scalar_if_unsafe(value, diagnostics, *, invalidate_output):
    if invalidate_output is None:
        return value
    tail_norm = jnp.asarray(getattr(diagnostics, "tail_norm", 0.0), dtype=jnp.float64)
    primal_residual = jnp.asarray(getattr(diagnostics, "primal_residual", 0.0), dtype=jnp.float64)
    converged = jnp.asarray(getattr(diagnostics, "converged", True))
    breakdown = jnp.asarray(getattr(diagnostics, "breakdown", False))
    metric = jnp.asarray(getattr(diagnostics, "convergence_metric", tail_norm), dtype=jnp.float64)
    ok = jnp.isfinite(tail_norm) & jnp.isfinite(primal_residual) & jnp.isfinite(metric) & converged & (~breakdown)
    while ok.ndim < value.ndim:
        ok = ok[..., None]
    return jnp.where(ok, value, invalidate_output(value))


def _invalidate_action_if_unsafe(value, diagnostics, *, invalidate_output):
    if invalidate_output is None:
        return value
    tail_norm = jnp.asarray(getattr(diagnostics, "tail_norm", 0.0), dtype=jnp.float64)
    primal_residual = jnp.asarray(getattr(diagnostics, "primal_residual", tail_norm), dtype=jnp.float64)
    converged = jnp.asarray(getattr(diagnostics, "converged", True))
    breakdown = jnp.asarray(getattr(diagnostics, "breakdown", False))
    metric = jnp.asarray(getattr(diagnostics, "convergence_metric", tail_norm), dtype=jnp.float64)
    ok = jnp.isfinite(tail_norm) & jnp.isfinite(primal_residual) & jnp.isfinite(metric) & converged & (~breakdown)
    while ok.ndim < value.ndim:
        ok = ok[..., None]
    return jnp.where(ok, value, invalidate_output(value))


__all__ = [
    "operator_apply_basic",
    "action_basic",
    "action_with_diagnostics_basic",
    "solve_action_basic",
    "solve_action_with_diagnostics_basic",
    "inverse_action_basic",
    "inverse_action_with_diagnostics_basic",
    "scalar_functional_basic",
    "vector_functional_basic",
    "scalar_functional_with_diagnostics_basic",
    "scalar_uncertainty_radius",
    "det_basic_from_logdet",
]
