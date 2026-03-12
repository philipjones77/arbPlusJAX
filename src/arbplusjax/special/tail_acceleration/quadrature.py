from __future__ import annotations

from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp

from .diagnostics import TailEvaluationDiagnostics


def finite_panel_tail_quadrature(
    integrand: Callable[[jax.Array], jax.Array],
    lower_limit: jax.Array,
    *,
    panel_width: float,
    max_panels: int,
    samples_per_panel: int,
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

    sample_offsets = jnp.linspace(0.0, panel_width, samples_per_panel, dtype=jnp.float64)
    panel_indices = jnp.arange(max_panels, dtype=jnp.int32)

    def panel_step(state, panel_index):
        total, _ = state
        start = lower + jnp.asarray(panel_index, dtype=jnp.float64) * jnp.asarray(panel_width, dtype=jnp.float64)
        grid = start + sample_offsets
        vals = jax.vmap(integrand)(grid)
        panel_val = jnp.trapezoid(vals, grid)
        last_panel_abs = jnp.trapezoid(jnp.abs(vals), grid)
        total = total + panel_val
        return (total, last_panel_abs), panel_val

    (total, last_panel_abs), _ = lax.scan(
        panel_step,
        (jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64)),
        panel_indices,
    )

    diagnostics = TailEvaluationDiagnostics(
        method=method,
        chunk_count=max_panels,
        panel_count=max_panels,
        recurrence_steps=0,
        estimated_tail_remainder=last_panel_abs,
        instability_flags=(),
        fallback_used=fallback_used,
        precision_warning=precision_warning,
        note=note,
    )
    return total, diagnostics
