from __future__ import annotations

"""Barnes G implementation family.

This module provides the repo's Barnes G implementation family and is tracked
explicitly for provenance so the implementation lineage can be located from the
source itself.

Provenance:
- classification: arb_like
- base_names: barnesg
- module lineage: repo Barnes G implementation used by the Arb-like surface
- naming policy: see docs/standards/function_naming.md
- registry report: see docs/status/reports/function_implementation_index.md
"""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import coeffs
from . import elementary as el


PROVENANCE = {
    "classification": "arb_like",
    "base_names": ("barnesg",),
    "module_lineage": "repo Barnes G implementation used by the Arb-like surface",
    "naming_policy": "docs/standards/function_naming.md",
    "registry_report": "docs/status/reports/function_implementation_index.md",
}

_LOG_A = jnp.float64(0.248754477)  # log Glaisher-Kinkelin constant
_LOG_2PI = el.LOG_TWO_PI

_BARNESG_B = jnp.asarray(
    [
        -1.0 / 30.0,  # B4
        1.0 / 42.0,  # B6
        -1.0 / 30.0,  # B8
        5.0 / 66.0,  # B10
        -691.0 / 2730.0,  # B12
        7.0 / 6.0,  # B14
        -3617.0 / 510.0,  # B16
        43867.0 / 798.0,  # B18
        -174611.0 / 330.0,  # B20
    ],
    dtype=jnp.float64,
)

_LANCZOS = coeffs.LANCZOS


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    z = jnp.asarray(z, dtype=cdt)
    z1 = z - jnp.asarray(1.0 + 0.0j, dtype=cdt)
    x = jnp.full_like(z, jnp.asarray(_LANCZOS[0] + 0.0j, dtype=cdt))

    def body(i, acc):
        return acc + jnp.asarray(_LANCZOS[i], dtype=cdt) / (z1 + jnp.asarray(i, dtype=rdt))

    x = lax.fori_loop(1, 9, body, x)
    t = z1 + jnp.asarray(7.5, dtype=rdt)
    return jnp.asarray(el.LOG_SQRT_TWO_PI, dtype=cdt) + (z1 + jnp.asarray(0.5, dtype=rdt)) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    z = jnp.asarray(z, dtype=cdt)

    def reflection(w):
        return jnp.asarray(el.LOG_PI, dtype=cdt) - jnp.log(jnp.sin(jnp.asarray(el.PI, dtype=cdt) * w)) - _complex_loggamma_lanczos(jnp.asarray(1.0, dtype=cdt) - w)

    return jnp.where(jnp.real(z) < 0.5, reflection(z), _complex_loggamma_lanczos(z))


def _log_barnesg_asymptotic(z: jax.Array) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    # Asymptotic expansion for log G(z), using formula for log G(w+1) with w = z-1
    w = jnp.asarray(z, dtype=cdt) - jnp.asarray(1.0 + 0.0j, dtype=cdt)
    logw = jnp.log(w)
    w2 = w * w
    term = (w2 / jnp.asarray(2.0, dtype=rdt) - jnp.asarray(1.0 / 12.0, dtype=rdt)) * logw - jnp.asarray(0.75, dtype=rdt) * w2 + jnp.asarray(0.5, dtype=rdt) * w * jnp.asarray(_LOG_2PI, dtype=cdt) + jnp.asarray(_LOG_A, dtype=cdt)

    def body(i, acc):
        k = jnp.asarray(i + 1, dtype=rdt)
        coeff = jnp.asarray(_BARNESG_B[i], dtype=cdt)
        return acc + coeff / (jnp.asarray(4.0, dtype=rdt) * k * (k + jnp.asarray(1.0, dtype=rdt)) * (w ** (jnp.asarray(2.0, dtype=rdt) * k)))

    return lax.fori_loop(0, _BARNESG_B.shape[0], body, term)


def _is_nonpositive_integer_real(x: jax.Array) -> jax.Array:
    return (x <= 0.0) & (jnp.abs(x - jnp.round(x)) < 1e-12)


@partial(jax.jit, static_argnames=("target",))
def log_barnesg(z: jax.Array, target: float = 5.0) -> jax.Array:
    cdt = el.complex_dtype_from(z)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    z = jnp.asarray(z, dtype=cdt)
    max_shift = 16
    n = jnp.maximum(jnp.asarray(0.0, dtype=rdt), jnp.ceil(jnp.asarray(target, dtype=rdt) - jnp.real(z)))
    n = jnp.minimum(n, jnp.asarray(max_shift, dtype=rdt))

    def body(i, state):
        zc, acc = state
        do = jnp.asarray(i, dtype=rdt) < n
        acc = jnp.where(do, acc - _complex_loggamma(zc), acc)
        zc = jnp.where(do, zc + jnp.asarray(1.0, dtype=rdt), zc)
        return zc, acc

    z_shift, acc = lax.fori_loop(0, max_shift, body, (z, jnp.asarray(0.0 + 0.0j, dtype=cdt)))
    return _log_barnesg_asymptotic(z_shift) + acc


@partial(jax.jit, static_argnames=("target",))
def barnesg(z: jax.Array, target: float = 5.0) -> jax.Array:
    return jnp.exp(log_barnesg(z, target=target))


@partial(jax.jit, static_argnames=("target",))
def barnesg_real(x: jax.Array, target: float = 5.0) -> jax.Array:
    x = el.as_real(x)
    pole = _is_nonpositive_integer_real(x)
    val = jnp.real(barnesg(x + 0.0j, target=target))
    return jnp.where(pole, jnp.nan, val)


@partial(jax.jit, static_argnames=("target",))
def barnesg_complex(z: jax.Array, target: float = 5.0) -> jax.Array:
    z = el.as_complex(z)
    pole = (jnp.imag(z) == 0.0) & _is_nonpositive_integer_real(jnp.real(z))
    val = barnesg(z, target=target)
    return jnp.where(pole, jnp.nan + 1j * jnp.nan, val)


__all__ = [
    "log_barnesg",
    "barnesg",
    "barnesg_real",
    "barnesg_complex",
]
