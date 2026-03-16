from __future__ import annotations

from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from .diagnostics import TailEvaluationDiagnostics


QuadratureRule = str


def _rule_name(rule: QuadratureRule) -> str:
    normalized = rule.strip().lower()
    aliases = {
        "trapz": "trapezoid",
        "trapezoidal": "trapezoid",
        "simpsons": "simpson",
        "gauss": "gauss_legendre",
        "legendre": "gauss_legendre",
    }
    return aliases.get(normalized, normalized)


def _simpson_point_count(samples_per_panel: int) -> int:
    n = max(int(samples_per_panel), 3)
    return n if n % 2 == 1 else n + 1


def _gauss_nodes_weights(order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(max(int(order), 2))
    return jnp.asarray(nodes, dtype=jnp.float64), jnp.asarray(weights, dtype=jnp.float64)


def _uniform_panel_eval(
    integrand: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    end: jax.Array,
    *,
    n_points: int,
    rule: str,
) -> tuple[jax.Array, jax.Array]:
    grid = jnp.linspace(start, end, n_points, dtype=jnp.float64)
    vals = jax.vmap(integrand)(grid)
    if rule == "trapezoid":
        return jnp.trapezoid(vals, grid), jnp.trapezoid(jnp.abs(vals), grid)
    h = (end - start) / jnp.asarray(n_points - 1, dtype=jnp.float64)
    odd_sum = jnp.sum(vals[1:-1:2])
    even_sum = jnp.sum(vals[2:-1:2])
    value = h / 3.0 * (vals[0] + vals[-1] + 4.0 * odd_sum + 2.0 * even_sum)
    abs_vals = jnp.abs(vals)
    odd_abs_sum = jnp.sum(abs_vals[1:-1:2])
    even_abs_sum = jnp.sum(abs_vals[2:-1:2])
    abs_value = h / 3.0 * (abs_vals[0] + abs_vals[-1] + 4.0 * odd_abs_sum + 2.0 * even_abs_sum)
    return value, abs_value


def _gauss_legendre_panel_eval(
    integrand: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    end: jax.Array,
    *,
    order: int,
) -> tuple[jax.Array, jax.Array]:
    nodes, weights = _gauss_nodes_weights(order)
    center = 0.5 * (start + end)
    half_width = 0.5 * (end - start)
    grid = center + half_width * nodes
    vals = jax.vmap(integrand)(grid)
    return half_width * jnp.sum(weights * vals), half_width * jnp.sum(weights * jnp.abs(vals))


def panel_quadrature(
    integrand: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    end: jax.Array,
    *,
    samples_per_panel: int,
    quadrature_rule: QuadratureRule = "simpson",
) -> tuple[jax.Array, jax.Array]:
    rule = _rule_name(quadrature_rule)
    if rule == "trapezoid":
        return _uniform_panel_eval(integrand, start, end, n_points=max(int(samples_per_panel), 2), rule=rule)
    if rule == "simpson":
        return _uniform_panel_eval(
            integrand,
            start,
            end,
            n_points=_simpson_point_count(samples_per_panel),
            rule=rule,
        )
    if rule == "gauss_legendre":
        return _gauss_legendre_panel_eval(integrand, start, end, order=max(int(samples_per_panel), 2))
    raise ValueError("quadrature_rule must be one of: trapezoid, simpson, gauss_legendre")


def _refined_panel_quadrature(
    integrand: Callable[[jax.Array], jax.Array],
    start: jax.Array,
    end: jax.Array,
    *,
    samples_per_panel: int,
    quadrature_rule: QuadratureRule,
) -> tuple[jax.Array, jax.Array]:
    rule = _rule_name(quadrature_rule)
    if rule == "trapezoid":
        refined_points = max(int(samples_per_panel) * 2 - 1, 3)
    elif rule == "simpson":
        refined_points = 2 * (_simpson_point_count(samples_per_panel) - 1) + 1
    elif rule == "gauss_legendre":
        refined_points = max(int(samples_per_panel) * 2, int(samples_per_panel) + 2)
    else:
        raise ValueError("quadrature_rule must be one of: trapezoid, simpson, gauss_legendre")
    return panel_quadrature(
        integrand,
        start,
        end,
        samples_per_panel=refined_points,
        quadrature_rule=rule,
    )


def finite_panel_tail_quadrature(
    integrand: Callable[[jax.Array], jax.Array],
    lower_limit: jax.Array,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    quadrature_rule: QuadratureRule = "simpson",
    method: str = "quadrature",
    fallback_used: bool = False,
    precision_warning: bool = False,
    note: str = "",
) -> tuple[jax.Array, TailEvaluationDiagnostics]:
    lower = jnp.asarray(lower_limit, dtype=jnp.float64)
    if max_panels <= 0:
        raise ValueError("max_panels must be > 0")
    if samples_per_panel < 2:
        raise ValueError("samples_per_panel must be >= 2")

    panel_indices = jnp.arange(max_panels, dtype=jnp.int32)

    def panel_step(state, panel_index):
        total, _, _ = state
        start = lower + jnp.asarray(panel_index, dtype=jnp.float64) * jnp.asarray(panel_width, dtype=jnp.float64)
        end = start + jnp.asarray(panel_width, dtype=jnp.float64)
        panel_val, panel_abs = panel_quadrature(
            integrand,
            start,
            end,
            samples_per_panel=samples_per_panel,
            quadrature_rule=quadrature_rule,
        )
        refined_val, _ = _refined_panel_quadrature(
            integrand,
            start,
            end,
            samples_per_panel=samples_per_panel,
            quadrature_rule=quadrature_rule,
        )
        panel_err = jnp.abs(refined_val - panel_val)
        total = total + refined_val
        return (total, panel_abs, panel_err), refined_val

    (total, last_panel_abs, last_panel_err), _ = lax.scan(
        panel_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        panel_indices,
    )

    diagnostics = TailEvaluationDiagnostics(
        method=method,
        chunk_count=max_panels,
        panel_count=max_panels,
        recurrence_steps=0,
        estimated_tail_remainder=jnp.maximum(last_panel_abs, last_panel_err),
        instability_flags=(),
        fallback_used=fallback_used,
        precision_warning=precision_warning,
        note=(note + f" quadrature_rule={_rule_name(quadrature_rule)}").strip(),
    )
    return total, diagnostics


def finite_interval_quadrature(
    integrand: Callable[[jax.Array], jax.Array],
    lower_limit: jax.Array,
    upper_limit: jax.Array,
    *,
    panel_count: int,
    samples_per_panel: int,
    quadrature_rule: QuadratureRule = "simpson",
    method: str = "quadrature",
    fallback_used: bool = False,
    precision_warning: bool = False,
    note: str = "",
) -> tuple[jax.Array, TailEvaluationDiagnostics]:
    lower = jnp.asarray(lower_limit, dtype=jnp.float64)
    upper = jnp.asarray(upper_limit, dtype=jnp.float64)
    if panel_count <= 0:
        raise ValueError("panel_count must be > 0")
    if samples_per_panel < 2:
        raise ValueError("samples_per_panel must be >= 2")

    panel_width = (upper - lower) / jnp.asarray(panel_count, dtype=jnp.float64)
    panel_indices = jnp.arange(panel_count, dtype=jnp.int32)

    def panel_step(state, panel_index):
        total, _, _ = state
        start = lower + jnp.asarray(panel_index, dtype=jnp.float64) * panel_width
        end = start + panel_width
        panel_val, panel_abs = panel_quadrature(
            integrand,
            start,
            end,
            samples_per_panel=samples_per_panel,
            quadrature_rule=quadrature_rule,
        )
        refined_val, _ = _refined_panel_quadrature(
            integrand,
            start,
            end,
            samples_per_panel=samples_per_panel,
            quadrature_rule=quadrature_rule,
        )
        panel_err = jnp.abs(refined_val - panel_val)
        total = total + refined_val
        return (total, panel_abs, panel_err), refined_val

    (value, last_panel_abs, last_panel_err), _ = lax.scan(
        panel_step,
        (
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ),
        panel_indices,
    )

    diagnostics = TailEvaluationDiagnostics(
        method=method,
        chunk_count=panel_count,
        panel_count=panel_count,
        recurrence_steps=0,
        estimated_tail_remainder=jnp.maximum(last_panel_abs, last_panel_err),
        instability_flags=(),
        fallback_used=fallback_used,
        precision_warning=precision_warning,
        note=(note + f" quadrature_rule={_rule_name(quadrature_rule)}").strip(),
    )
    return value, diagnostics
