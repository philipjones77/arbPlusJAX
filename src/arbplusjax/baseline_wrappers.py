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
    if mode == "baseline":
        mode = "basic"
    checks.check_in_set(mode, ("basic", "rigorous", "adaptive"), "baseline_wrappers.mode")
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
    if mode == "basic":
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
    if mode == "basic":
        return base_fn(x, prec_bits=prec_bits)
    if mode == "rigorous":
        return rig_fn(x, prec_bits=prec_bits)
    if mode == "adaptive":
        return adapt_fn(x, prec_bits=prec_bits)
    return base_fn(x, prec_bits=prec_bits)


def _dispatch_real_bivariate(
    mode: str,
    base_fn,
    rig_fn,
    adapt_fn,
    a: jax.Array,
    b: jax.Array,
    prec_bits: int,
) -> jax.Array:
    _mode_ok(mode)
    if mode == "basic":
        return base_fn(a, b, prec_bits=prec_bits)
    if mode == "rigorous":
        return rig_fn(a, b, prec_bits=prec_bits)
    if mode == "adaptive":
        return adapt_fn(a, b, prec_bits=prec_bits)
    return base_fn(a, b, prec_bits=prec_bits)


def _dispatch_real_trivariate(
    mode: str,
    base_fn,
    rig_fn,
    adapt_fn,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    prec_bits: int,
) -> jax.Array:
    _mode_ok(mode)
    if mode == "basic":
        return base_fn(a, b, c, prec_bits=prec_bits)
    if mode == "rigorous":
        return rig_fn(a, b, c, prec_bits=prec_bits)
    if mode == "adaptive":
        return adapt_fn(a, b, c, prec_bits=prec_bits)
    return base_fn(a, b, c, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_exp_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_exp_prec, ball_wrappers.arb_ball_exp, ball_wrappers.arb_ball_exp_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_log_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_log_prec, ball_wrappers.arb_ball_log, ball_wrappers.arb_ball_log_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sqrt_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_sqrt_prec, ball_wrappers.arb_ball_sqrt, ball_wrappers.arb_ball_sqrt_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sin_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_sin_prec, ball_wrappers.arb_ball_sin, ball_wrappers.arb_ball_sin_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_cos_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_cos_prec, ball_wrappers.arb_ball_cos, ball_wrappers.arb_ball_cos_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_tan_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_tan_prec, ball_wrappers.arb_ball_tan, ball_wrappers.arb_ball_tan_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sinh_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_sinh_prec, ball_wrappers.arb_ball_sinh, ball_wrappers.arb_ball_sinh_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_cosh_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_cosh_prec, ball_wrappers.arb_ball_cosh, ball_wrappers.arb_ball_cosh_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_tanh_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_tanh_prec, ball_wrappers.arb_ball_tanh, ball_wrappers.arb_ball_tanh_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_abs_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_abs_prec, arb_core.arb_abs_prec, arb_core.arb_abs_prec, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_add_mp(
    x: jax.Array,
    y: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real_bivariate(mode, arb_core.arb_add_prec, arb_core.arb_add_prec, arb_core.arb_add_prec, x, y, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sub_mp(
    x: jax.Array,
    y: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real_bivariate(mode, arb_core.arb_sub_prec, arb_core.arb_sub_prec, arb_core.arb_sub_prec, x, y, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_mul_mp(
    x: jax.Array,
    y: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real_bivariate(mode, arb_core.arb_mul_prec, arb_core.arb_mul_prec, arb_core.arb_mul_prec, x, y, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_div_mp(
    x: jax.Array,
    y: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real_bivariate(mode, arb_core.arb_div_prec, arb_core.arb_div_prec, arb_core.arb_div_prec, x, y, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_inv_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_inv_prec, arb_core.arb_inv_prec, arb_core.arb_inv_prec, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_fma_mp(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real_trivariate(mode, arb_core.arb_fma_prec, arb_core.arb_fma_prec, arb_core.arb_fma_prec, x, y, z, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_log1p_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_log1p_prec, arb_core.arb_log1p_prec, arb_core.arb_log1p_prec, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_expm1_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_expm1_prec, arb_core.arb_expm1_prec, arb_core.arb_expm1_prec, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_sin_cos_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None):
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, arb_core.arb_sin_cos_prec, arb_core.arb_sin_cos_prec, arb_core.arb_sin_cos_prec, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_gamma_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_gamma_prec, ball_wrappers.arb_ball_gamma, ball_wrappers.arb_ball_gamma_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_erf_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_erf_prec, ball_wrappers.arb_ball_erf, ball_wrappers.arb_ball_erf_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_erfc_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_erfc_prec, ball_wrappers.arb_ball_erfc, ball_wrappers.arb_ball_erfc_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_erfi_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_erfi_prec, ball_wrappers.arb_ball_erfi, ball_wrappers.arb_ball_erfi_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_exp_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_exp_prec, ball_wrappers.acb_ball_exp, ball_wrappers.acb_ball_exp_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_log_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_log_prec, ball_wrappers.acb_ball_log, ball_wrappers.acb_ball_log_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_sin_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_sin_prec, ball_wrappers.acb_ball_sin, ball_wrappers.acb_ball_sin_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_barnesg_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_real(mode, hypgeom.arb_hypgeom_barnesg_prec, ball_wrappers.arb_ball_barnesg, ball_wrappers.arb_ball_barnesg_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_barnesg_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, hypgeom.acb_hypgeom_barnesg_prec, ball_wrappers.acb_ball_barnesg, ball_wrappers.acb_ball_barnesg_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def acb_gamma_mp(x: jax.Array, mode: str = "basic", prec_bits: int | None = None, dps: int | None = None) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    return _dispatch_complex(mode, acb_core.acb_gamma, ball_wrappers.acb_ball_gamma, ball_wrappers.acb_ball_gamma_adaptive, x, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_bessel_j_mp(
    nu: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    base = partial(hypgeom.arb_hypgeom_bessel_j_prec, mode="midpoint")
    return _dispatch_real_bivariate(mode, base, ball_wrappers.arb_ball_bessel_j, ball_wrappers.arb_ball_bessel_j_adaptive, nu, z, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_bessel_y_mp(
    nu: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    base = partial(hypgeom.arb_hypgeom_bessel_y_prec, mode="midpoint")
    return _dispatch_real_bivariate(mode, base, ball_wrappers.arb_ball_bessel_y, ball_wrappers.arb_ball_bessel_y_adaptive, nu, z, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_bessel_i_mp(
    nu: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    base = partial(hypgeom.arb_hypgeom_bessel_i_prec, mode="midpoint")
    return _dispatch_real_bivariate(mode, base, ball_wrappers.arb_ball_bessel_i, ball_wrappers.arb_ball_bessel_i_adaptive, nu, z, pb)


@partial(jax.jit, static_argnames=("mode", "prec_bits", "dps"))
def arb_bessel_k_mp(
    nu: jax.Array,
    z: jax.Array,
    mode: str = "basic",
    prec_bits: int | None = None,
    dps: int | None = None,
) -> jax.Array:
    pb = _prec_bits(dps, prec_bits)
    base = partial(hypgeom.arb_hypgeom_bessel_k_prec, mode="midpoint")
    return _dispatch_real_bivariate(mode, base, ball_wrappers.arb_ball_bessel_k, ball_wrappers.arb_ball_bessel_k_adaptive, nu, z, pb)


__all__ = [
    "arb_exp_mp",
    "arb_log_mp",
    "arb_sqrt_mp",
    "arb_sin_mp",
    "arb_cos_mp",
    "arb_tan_mp",
    "arb_sinh_mp",
    "arb_cosh_mp",
    "arb_tanh_mp",
    "arb_abs_mp",
    "arb_add_mp",
    "arb_sub_mp",
    "arb_mul_mp",
    "arb_div_mp",
    "arb_inv_mp",
    "arb_fma_mp",
    "arb_log1p_mp",
    "arb_expm1_mp",
    "arb_sin_cos_mp",
    "arb_gamma_mp",
    "arb_erf_mp",
    "arb_erfc_mp",
    "arb_erfi_mp",
    "acb_exp_mp",
    "acb_log_mp",
    "acb_sin_mp",
    "acb_gamma_mp",
    "arb_barnesg_mp",
    "acb_barnesg_mp",
    "arb_bessel_j_mp",
    "arb_bessel_y_mp",
    "arb_bessel_i_mp",
    "arb_bessel_k_mp",
]
