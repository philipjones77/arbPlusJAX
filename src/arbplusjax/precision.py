from __future__ import annotations

from contextlib import contextmanager
from math import ceil, log10

import jax
import jax.numpy as jnp

_DPS = 50
_PREC_BITS = 53


def enable_jax_x64() -> None:
    jax.config.update("jax_enable_x64", True)

def set_jax_x64(enabled: bool) -> None:
    jax.config.update("jax_enable_x64", bool(enabled))


def jax_x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def dps_to_bits(dps: int) -> int:
    return int(ceil(dps * log10(10) / log10(2)))


def set_dps(dps: int) -> None:
    global _DPS, _PREC_BITS
    _DPS = int(dps)
    _PREC_BITS = dps_to_bits(_DPS)


def set_prec_bits(prec_bits: int) -> None:
    global _DPS, _PREC_BITS
    _PREC_BITS = int(prec_bits)
    _DPS = int(ceil(_PREC_BITS * log10(2) / log10(10)))


def get_dps() -> int:
    return _DPS


def get_prec_bits() -> int:
    return _PREC_BITS


@contextmanager
def workdps(dps: int):
    old = _DPS
    set_dps(dps)
    try:
        yield
    finally:
        set_dps(old)


@contextmanager
def workprec(prec_bits: int):
    old = _PREC_BITS
    set_prec_bits(prec_bits)
    try:
        yield
    finally:
        set_prec_bits(old)


@contextmanager
def jax_x64_context(enabled: bool = True):
    old = jax_x64_enabled()
    set_jax_x64(enabled)
    try:
        yield
    finally:
        set_jax_x64(old)


def eps_from_dps(dps: int | None = None) -> jnp.ndarray:
    bits = dps_to_bits(_DPS if dps is None else dps)
    return jnp.exp2(-jnp.float64(bits))


__all__ = [
    "enable_jax_x64",
    "set_jax_x64",
    "jax_x64_enabled",
    "jax_x64_context",
    "dps_to_bits",
    "set_dps",
    "set_prec_bits",
    "get_dps",
    "get_prec_bits",
    "workdps",
    "workprec",
    "eps_from_dps",
]
