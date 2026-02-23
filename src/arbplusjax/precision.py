from __future__ import annotations

from contextlib import contextmanager
from math import ceil, log10

import jax.numpy as jnp

_DPS = 50
_PREC_BITS = 53


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


def eps_from_dps(dps: int | None = None) -> jnp.ndarray:
    bits = dps_to_bits(_DPS if dps is None else dps)
    return jnp.exp2(-jnp.float64(bits))
