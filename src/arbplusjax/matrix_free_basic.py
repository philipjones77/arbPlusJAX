from __future__ import annotations

import jax

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
    **kwargs,
):
    value, diagnostics = point_action_with_diagnostics_fn(operator, x, *args, **kwargs)
    return round_output(value, prec_bits), diagnostics


def solve_action_basic(point_solve_fn, operator, b: jax.Array, *, round_output, prec_bits: int, **kwargs):
    return round_output(point_solve_fn(operator, b, **kwargs), prec_bits)


def solve_action_with_diagnostics_basic(
    point_solve_with_diagnostics_fn,
    operator,
    b: jax.Array,
    *,
    round_output,
    prec_bits: int,
    **kwargs,
):
    value, diagnostics = point_solve_with_diagnostics_fn(operator, b, **kwargs)
    return round_output(value, prec_bits), diagnostics


def inverse_action_basic(point_inverse_fn, operator, x: jax.Array, *, round_output, prec_bits: int, **kwargs):
    return round_output(point_inverse_fn(operator, x, **kwargs), prec_bits)


def inverse_action_with_diagnostics_basic(
    point_inverse_with_diagnostics_fn,
    operator,
    x: jax.Array,
    *,
    round_output,
    prec_bits: int,
    **kwargs,
):
    value, diagnostics = point_inverse_with_diagnostics_fn(operator, x, **kwargs)
    return round_output(value, prec_bits), diagnostics


def scalar_functional_basic(point_scalar_fn, *args, lift_scalar, round_output, prec_bits: int, **kwargs):
    return round_output(lift_scalar(point_scalar_fn(*args, **kwargs)), prec_bits)


def scalar_functional_with_diagnostics_basic(
    point_scalar_with_diagnostics_fn,
    *args,
    lift_scalar,
    round_output,
    prec_bits: int,
    **kwargs,
):
    value, diagnostics = point_scalar_with_diagnostics_fn(*args, **kwargs)
    return round_output(lift_scalar(value), prec_bits), diagnostics


def det_basic_from_logdet(point_logdet_fn, *args, lift_scalar, round_output, prec_bits: int, **kwargs):
    logdet = point_logdet_fn(*args, **kwargs)
    return round_output(lift_scalar(matrix_free_core.det_from_logdet(logdet)), prec_bits)


__all__ = [
    "operator_apply_basic",
    "action_basic",
    "action_with_diagnostics_basic",
    "solve_action_basic",
    "solve_action_with_diagnostics_basic",
    "inverse_action_basic",
    "inverse_action_with_diagnostics_basic",
    "scalar_functional_basic",
    "scalar_functional_with_diagnostics_basic",
    "det_basic_from_logdet",
]
