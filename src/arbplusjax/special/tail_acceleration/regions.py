from __future__ import annotations

from .recurrence import TailRatioRecurrence


def choose_tail_method(
    *,
    requested_method: str,
    has_recurrence: bool,
    decay_rate: float | None,
    oscillation_level: float | None,
    derivatives_required: bool,
) -> str:
    if requested_method == "mpfallback":
        requested_method = "high_precision_refine"
    if requested_method != "auto":
        return requested_method
    if has_recurrence and not derivatives_required and (oscillation_level or 0.0) <= 0.5:
        return "recurrence"
    if (oscillation_level or 0.0) > 1.0:
        return "wynn"
    if (decay_rate or 0.0) >= 1.0:
        return "quadrature"
    return "aitken"


def recurrence_is_available(recurrence: TailRatioRecurrence | None) -> bool:
    return recurrence is not None
