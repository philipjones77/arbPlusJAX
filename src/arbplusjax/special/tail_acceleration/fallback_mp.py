from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .diagnostics import TailEvaluationDiagnostics
from .quadrature import finite_panel_tail_quadrature


def mp_tail_integral_fallback(
    integrand: Callable[[float], float],
    lower_limit: float,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
    quadrature_rule: str = "simpson",
) -> tuple[float, TailEvaluationDiagnostics]:
    note = "Pure-JAX high-precision refinement fallback only; true arbitrary precision is still pending."

    coarse_value, coarse_diag = finite_panel_tail_quadrature(
        integrand,
        lower_limit,
        panel_width=panel_width,
        max_panels=max(max_panels * 2, 64),
        samples_per_panel=max(samples_per_panel * 2, 64),
        quadrature_rule="simpson",
        method="high_precision_refine",
        fallback_used=True,
        precision_warning=True,
        note=note,
    )
    refined_value, refined_diag = finite_panel_tail_quadrature(
        integrand,
        lower_limit,
        panel_width=panel_width / 2.0,
        max_panels=max(max_panels * 4, 128),
        samples_per_panel=max(samples_per_panel * 4, 128),
        quadrature_rule="gauss_legendre",
        method="high_precision_refine",
        fallback_used=True,
        precision_warning=True,
        note=note,
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

    diagnostics = TailEvaluationDiagnostics(
        method="high_precision_refine",
        chunk_count=coarse_diag.chunk_count + refined_diag.chunk_count,
        panel_count=coarse_diag.panel_count + refined_diag.panel_count,
        recurrence_steps=0,
        estimated_tail_remainder=float(estimated_remainder),
        instability_flags=tuple(instability_flags),
        fallback_used=True,
        precision_warning=True,
        note=note,
    )
    return float(refined_value), diagnostics
