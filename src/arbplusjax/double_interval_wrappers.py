from __future__ import annotations

import jax
import jax.numpy as jnp

from . import double_interval as di
from . import wrappers_common as wc
from . import checks

jax.config.update("jax_enable_x64", True)


def _resolve_prec_bits(dps: int | None, prec_bits: int | None) -> int:
    return wc.resolve_prec_bits(dps, prec_bits)


def _inflate_interval(x: jax.Array, prec_bits: int, adaptive: bool) -> jax.Array:
    return wc.inflate_interval(x, prec_bits, adaptive)


def fast_add_mode(x: jax.Array, y: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_add(x, y)
    if impl == "rigorous":
        return di.fast_add_prec(x, y, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_add_prec(x, y, prec_bits=pb), pb, adaptive=True)
    return di.fast_add(x, y)


def fast_sub_mode(x: jax.Array, y: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_sub(x, y)
    if impl == "rigorous":
        return di.fast_sub_prec(x, y, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_sub_prec(x, y, prec_bits=pb), pb, adaptive=True)
    return di.fast_sub(x, y)


def fast_mul_mode(x: jax.Array, y: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_mul(x, y)
    if impl == "rigorous":
        return di.fast_mul_prec(x, y, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_mul_prec(x, y, prec_bits=pb), pb, adaptive=True)
    return di.fast_mul(x, y)


def fast_div_mode(x: jax.Array, y: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_div(x, y)
    if impl == "rigorous":
        return di.fast_div_prec(x, y, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_div_prec(x, y, prec_bits=pb), pb, adaptive=True)
    return di.fast_div(x, y)


def fast_sqr_mode(x: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_sqr(x)
    if impl == "rigorous":
        return di.fast_sqr_prec(x, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_sqr_prec(x, prec_bits=pb), pb, adaptive=True)
    return di.fast_sqr(x)


def fast_log_nonnegative_mode(x: jax.Array, impl: str = "baseline", dps: int | None = None, prec_bits: int | None = None) -> jax.Array:
    pb = _resolve_prec_bits(dps, prec_bits)
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "double_interval_wrappers.impl")
    if impl == "baseline":
        return di.fast_log_nonnegative(x)
    if impl == "rigorous":
        return di.fast_log_nonnegative_prec(x, prec_bits=pb)
    if impl == "adaptive":
        return _inflate_interval(di.fast_log_nonnegative_prec(x, prec_bits=pb), pb, adaptive=True)
    return di.fast_log_nonnegative(x)


__all__ = [
    "fast_add_mode",
    "fast_sub_mode",
    "fast_mul_mode",
    "fast_div_mode",
    "fast_sqr_mode",
    "fast_log_nonnegative_mode",
]
