from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
from jax import lax
import jax.numpy as jnp

from .diagnostics import TailEvaluationDiagnostics
from .fallback_mp import mp_tail_integral_fallback
from .quadrature import finite_panel_tail_quadrature
from .recurrence import TailRatioRecurrence, ratio_recurrence_estimate
from .regions import choose_tail_method, recurrence_is_available
from .sequence import aitken_delta_squared, wynn_epsilon


@dataclass(frozen=True)
class TailDerivativeMetadata:
    argument_derivative: bool = False
    lower_limit_derivative: bool = False
    parameter_derivative: bool = False
    note: str = ""


@dataclass(frozen=True)
class TailRegimeMetadata:
    decay_rate: float | None = None
    oscillation_level: float | None = None
    near_singularity: bool = False
    cancellation_risk: bool = False
    note: str = ""


@dataclass(frozen=True)
class TailIntegralProblem:
    integrand: Callable[[jax.Array], jax.Array]
    lower_limit: float | jax.Array
    panel_width: float = 0.25
    max_panels: int = 128
    samples_per_panel: int = 32
    recurrence: TailRatioRecurrence | None = None
    derivative_metadata: TailDerivativeMetadata = TailDerivativeMetadata()
    regime_metadata: TailRegimeMetadata = TailRegimeMetadata()
    name: str = "tail_integral"


def evaluate_tail_integral(
    problem: TailIntegralProblem,
    *,
    method: str = "auto",
    return_diagnostics: bool = False,
) -> jax.Array | tuple[jax.Array, TailEvaluationDiagnostics]:
    chosen_method = choose_tail_method(
        requested_method=method,
        has_recurrence=recurrence_is_available(problem.recurrence),
        decay_rate=problem.regime_metadata.decay_rate,
        oscillation_level=problem.regime_metadata.oscillation_level,
        derivatives_required=problem.derivative_metadata.argument_derivative
        or problem.derivative_metadata.lower_limit_derivative
        or problem.derivative_metadata.parameter_derivative,
    )

    if chosen_method == "quadrature":
        value, diagnostics = finite_panel_tail_quadrature(
            problem.integrand,
            problem.lower_limit,
            panel_width=problem.panel_width,
            max_panels=problem.max_panels,
            samples_per_panel=problem.samples_per_panel,
        )
    elif chosen_method == "mpfallback":
        value, diagnostics = mp_tail_integral_fallback(
            problem.integrand,
            problem.lower_limit,
            panel_width=problem.panel_width,
            max_panels=problem.max_panels,
            samples_per_panel=problem.samples_per_panel,
        )
        value = jnp.asarray(value, dtype=jnp.float64)
    elif chosen_method == "recurrence":
        if problem.recurrence is None:
            raise ValueError("recurrence method requested but no recurrence metadata is attached.")
        value = jnp.asarray(ratio_recurrence_estimate(problem.recurrence, n_terms=problem.max_panels), dtype=jnp.float64)
        diagnostics = TailEvaluationDiagnostics(
            method="recurrence",
            chunk_count=0,
            panel_count=0,
            recurrence_steps=problem.max_panels,
            estimated_tail_remainder=jnp.abs(value) * jnp.asarray(1e-8, dtype=jnp.float64),
            instability_flags=(),
            fallback_used=False,
            precision_warning=False,
            note=problem.recurrence.note,
        )
    elif chosen_method in {"aitken", "wynn"}:
        base_value, base_diagnostics = finite_panel_tail_quadrature(
            problem.integrand,
            problem.lower_limit,
            panel_width=problem.panel_width,
            max_panels=problem.max_panels,
            samples_per_panel=problem.samples_per_panel,
            method=chosen_method,
            note="Sequence acceleration scaffold built from cumulative panel sums.",
        )
        partials = _panel_partial_sums(problem)
        accelerated = aitken_delta_squared(partials) if chosen_method == "aitken" else wynn_epsilon(partials)
        value = accelerated[-1] if accelerated.shape[0] else base_value
        diagnostics = TailEvaluationDiagnostics(
            method=chosen_method,
            chunk_count=base_diagnostics.chunk_count,
            panel_count=base_diagnostics.panel_count,
            recurrence_steps=0,
            estimated_tail_remainder=base_diagnostics.estimated_tail_remainder,
            instability_flags=base_diagnostics.instability_flags,
            fallback_used=False,
            precision_warning=False,
            note=base_diagnostics.note,
        )
    else:
        raise ValueError("method must be one of: auto, quadrature, aitken, wynn, recurrence, mpfallback")

    if return_diagnostics:
        return value, diagnostics
    return value


def _panel_partial_sums(problem: TailIntegralProblem) -> jnp.ndarray:
    lower = jnp.asarray(problem.lower_limit, dtype=jnp.float64)
    sample_offsets = jnp.linspace(0.0, problem.panel_width, problem.samples_per_panel, dtype=jnp.float64)
    panel_indices = jnp.arange(problem.max_panels, dtype=jnp.int32)

    def panel_step(total, panel_index):
        start = lower + jnp.asarray(panel_index, dtype=jnp.float64) * jnp.asarray(problem.panel_width, dtype=jnp.float64)
        grid = start + sample_offsets
        vals = jax.vmap(problem.integrand)(grid)
        total = total + jnp.trapezoid(vals, grid)
        return total, total

    _, partials = lax.scan(panel_step, jnp.asarray(0.0, dtype=jnp.float64), panel_indices)
    return partials
