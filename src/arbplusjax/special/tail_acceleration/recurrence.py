from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class TailRatioRecurrence:
    a0: float = 0.0
    a1: float = 1.0
    b0: float = 1.0
    b1: float = 1.0
    alpha: Callable[[int], float] | None = None
    beta: Callable[[int], float] | None = None
    gamma: Callable[[int], float] | None = None
    delta: Callable[[int], float] | None = None
    a_init: tuple[float, ...] = ()
    b_init: tuple[float, ...] = ()
    a_coeffs: Callable[[int], tuple[float, ...]] | None = None
    b_coeffs: Callable[[int], tuple[float, ...]] | None = None
    order: int = 2
    note: str = ""


@dataclass(frozen=True)
class TailRatioRecurrenceDiagnostics:
    recurrence_steps: int
    estimated_remainder: float
    instability_flags: tuple[str, ...] = ()
    note: str = ""


def ratio_recurrence_terms(spec: TailRatioRecurrence, n_terms: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    order = _validate_order(spec)
    a_terms = _initial_terms(spec.a_init, (spec.a0, spec.a1), order)
    b_terms = _initial_terms(spec.b_init, (spec.b0, spec.b1), order)
    if n_terms < order:
        raise ValueError(f"n_terms must be >= recurrence order ({order}).")

    for n in range(order - 1, n_terms - 1):
        a_next = _advance_terms(n, a_terms, order, spec.a_coeffs, spec.alpha, spec.beta, "a")
        b_next = _advance_terms(n, b_terms, order, spec.b_coeffs, spec.gamma, spec.delta, "b")
        a_terms.append(jnp.asarray(a_next, dtype=jnp.float64))
        b_terms.append(jnp.asarray(b_next, dtype=jnp.float64))
    return jnp.stack(a_terms), jnp.stack(b_terms)


def ratio_recurrence_estimate(
    spec: TailRatioRecurrence,
    n_terms: int,
    *,
    return_diagnostics: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, TailRatioRecurrenceDiagnostics]:
    a_terms, b_terms = ratio_recurrence_terms(spec, n_terms=n_terms)
    ratio = _safe_ratio(a_terms[-1], b_terms[-1])
    diagnostics = ratio_recurrence_diagnostics(a_terms, b_terms, note=spec.note)
    if return_diagnostics:
        return ratio, diagnostics
    return ratio


def ratio_recurrence_diagnostics(
    a_terms: jnp.ndarray,
    b_terms: jnp.ndarray,
    *,
    note: str = "",
) -> TailRatioRecurrenceDiagnostics:
    ratios = _safe_ratio(a_terms, b_terms)
    instability_flags: list[str] = []
    if bool(jnp.any(jnp.abs(b_terms) < jnp.asarray(1e-14, dtype=jnp.float64))):
        instability_flags.append("small_denominator")
    if bool(jnp.any(~jnp.isfinite(ratios))):
        instability_flags.append("nonfinite_ratio")

    if ratios.shape[0] >= 3:
        last_delta = jnp.abs(ratios[-1] - ratios[-2])
        prev_delta = jnp.abs(ratios[-2] - ratios[-3])
        estimated_remainder = jnp.maximum(last_delta, jnp.abs(ratios[-1] - ratios[-3]))
        if bool(last_delta > 1.25 * prev_delta + jnp.asarray(1e-14, dtype=jnp.float64)):
            instability_flags.append("ratio_nonconverged")
    elif ratios.shape[0] >= 2:
        estimated_remainder = jnp.abs(ratios[-1] - ratios[-2])
    else:
        estimated_remainder = jnp.abs(ratios[-1]) * jnp.asarray(1e-8, dtype=jnp.float64)

    return TailRatioRecurrenceDiagnostics(
        recurrence_steps=int(a_terms.shape[0]),
        estimated_remainder=estimated_remainder,
        instability_flags=tuple(instability_flags),
        note=note,
    )


def _validate_order(spec: TailRatioRecurrence) -> int:
    order = int(spec.order)
    if order < 2:
        raise ValueError("recurrence order must be >= 2")
    if order not in {2, 4}:
        raise ValueError("recurrence order must be one of: 2, 4")
    if order == 4 and spec.a_coeffs is None:
        raise ValueError("order-4 recurrences require a_coeffs and b_coeffs callables.")
    if order == 4 and spec.b_coeffs is None:
        raise ValueError("order-4 recurrences require a_coeffs and b_coeffs callables.")
    return order


def _initial_terms(init: tuple[float, ...], legacy_init: tuple[float, float], order: int) -> list[jnp.ndarray]:
    if init:
        if len(init) != order:
            raise ValueError(f"initial history length must equal recurrence order ({order}).")
        values = init
    elif order == 2:
        values = legacy_init
    else:
        raise ValueError(f"order-{order} recurrences require explicit initial history.")
    return [jnp.asarray(value, dtype=jnp.float64) for value in values]


def _advance_terms(
    n: int,
    history: list[jnp.ndarray],
    order: int,
    coeffs_fn: Callable[[int], tuple[float, ...]] | None,
    coeff0: Callable[[int], float] | None,
    coeff1: Callable[[int], float] | None,
    label: str,
) -> jnp.ndarray:
    if coeffs_fn is not None:
        coeffs = coeffs_fn(n)
    else:
        if order != 2 or coeff0 is None or coeff1 is None:
            raise ValueError(f"missing recurrence coefficients for {label}-sequence.")
        coeffs = (coeff0(n), coeff1(n))
    if len(coeffs) != order:
        raise ValueError(f"{label}-sequence coefficient length must equal recurrence order ({order}).")
    recent = history[-order:]
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for coeff, term in zip(coeffs, reversed(recent)):
        total = total + jnp.asarray(coeff, dtype=jnp.float64) * jnp.asarray(term, dtype=jnp.float64)
    return total


def _safe_ratio(a, b):
    a_val = jnp.asarray(a, dtype=jnp.float64)
    b_val = jnp.asarray(b, dtype=jnp.float64)
    eps = jnp.asarray(1e-30, dtype=jnp.float64)
    safe_b = jnp.where(jnp.abs(b_val) > eps, b_val, jnp.sign(b_val) * eps + (b_val == 0.0) * eps)
    return a_val / safe_b
