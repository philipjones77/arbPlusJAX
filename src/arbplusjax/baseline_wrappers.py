from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import arb_core, acb_core, hypgeom
from . import precision
from . import ball_wrappers
from . import checks

jax.config.update("jax_enable_x64", True)


def _mode_ok(mode: str) -> str:
    checks.check_in_set(mode, ("baseline", "rigorous", "adaptive"), "baseline_wrappers.mode")
    return mode


def _prec_bits(dps: int | None, prec_bits: int | None) -> int:
    if prec_bits is not None:
        return int(prec_bits)
    if dps is not None:
        return precision.dps_to_bits(dps)
    return precision.get_prec_bits()


def _dispatch_real(
    mode: str,
    base_fn,
    rig_fn,
    adapt_fn,
    x: jax.Array,
    prec_bits: int,
) -> jax.Array:
    _mode_ok(mode)
    if mode == "baseline":
        return base_fn(x, prec_bits=prec_bits)
    if mode == "rigorous":
        return rig_fn(x, prec_bits=prec_bits)
    if mode == "adaptive":
        return adapt_fn(x, prec_bits=prec_bits)
    return base_fn(x, prec_bits=prec_bits)


def _dispatch_complex(
    mode: str,
    base_fn,
    rig_fn,
    adapt_fn,
    x: jax.Array,
    prec_bits: int,
) -> jax.Array:
    _mode_ok(mode)
    if mode == "baseline":
        return base_fn(x, prec_bits=prec_bits)
    if mode == "rigorous":
        return rig_fn(x, prec_bits=prec_bits)
    if mode == "adaptive":
        return adapt_fn(x, prec_bits=prec_bits)
    return base_fn(x, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_exp_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_exp_prec, ball_wrappers.arb_ball_exp, ball_wrappers.arb_ball_exp_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_log_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_log_prec, ball_wrappers.arb_ball_log, ball_wrappers.arb_ball_log_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sin_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_sin_prec, ball_wrappers.arb_ball_sin, ball_wrappers.arb_ball_sin_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_gamma_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_gamma_prec, ball_wrappers.arb_ball_gamma, ball_wrappers.arb_ball_gamma_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_exp_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_exp_prec, ball_wrappers.acb_ball_exp, ball_wrappers.acb_ball_exp_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_log_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_log_prec, ball_wrappers.acb_ball_log, ball_wrappers.acb_ball_log_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_sin_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_sin_prec, ball_wrappers.acb_ball_sin, ball_wrappers.acb_ball_sin_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_barnesg_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_barnesg_prec, ball_wrappers.arb_ball_barnesg, ball_wrappers.arb_ball_barnesg_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_barnesg_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, hypgeom.acb_hypgeom_barnesg_prec, ball_wrappers.acb_ball_barnesg, ball_wrappers.acb_ball_barnesg_adaptive, x, pb)


def acb_gamma_mp(x: jax.Array, mode: str = "baseline", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_gamma, ball_wrappers.acb_ball_gamma, ball_wrappers.acb_ball_gamma_adaptive, x, pb)


__all__ = [
    "arb_exp_mp",
    "arb_log_mp",
    "arb_sin_mp",
    "arb_gamma_mp",
    "acb_exp_mp",
    "acb_log_mp",
    "acb_sin_mp",
    "acb_gamma_mp",
]
