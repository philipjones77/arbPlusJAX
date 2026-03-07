from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from . import barnesg
from . import elementary as el

jax.config.update("jax_enable_x64", True)

_BESSEL_REAL_TERMS = 80
_BESSEL_COMPLEX_TERMS = 60


def _real_dtype(*xs: jax.Array) -> jnp.dtype:
    dt = jnp.result_type(*[jnp.asarray(x).dtype for x in xs])
    return jnp.dtype(jnp.float32) if dt == jnp.dtype(jnp.float32) else jnp.dtype(jnp.float64)


def _complex_dtype(*xs: jax.Array) -> jnp.dtype:
    return el.complex_dtype_from(*xs)


def real_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    sign = jnp.asarray(sign, dtype=rdt)
    half = jnp.asarray(0.5, dtype=rdt) * z
    quarter = jnp.asarray(0.25, dtype=rdt)
    term0 = jnp.power(half, nu) / jnp.exp(lax.lgamma(nu + jnp.asarray(1.0, dtype=rdt)))
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=rdt)
        den = k1 * (k1 + nu)
        num = quarter * sign * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, _BESSEL_REAL_TERMS - 1, body, (term0, sum0))
    return s


def real_bessel_asym_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    quarter = jnp.asarray(0.25, dtype=rdt)
    return jnp.sqrt(two / (jnp.asarray(el.PI, dtype=rdt) * z)) * jnp.cos(z - jnp.asarray(el.HALF_PI, dtype=rdt) * nu - quarter * jnp.asarray(el.PI, dtype=rdt))


def real_bessel_asym_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    quarter = jnp.asarray(0.25, dtype=rdt)
    return jnp.sqrt(two / (jnp.asarray(el.PI, dtype=rdt) * z)) * jnp.sin(z - jnp.asarray(el.HALF_PI, dtype=rdt) * nu - quarter * jnp.asarray(el.PI, dtype=rdt))


def real_bessel_asym_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    z = jnp.asarray(z, dtype=rdt)
    return jnp.exp(z) / jnp.sqrt(jnp.asarray(el.TWO_PI, dtype=rdt) * z)


def real_bessel_asym_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    z = jnp.asarray(z, dtype=rdt)
    two = jnp.asarray(2.0, dtype=rdt)
    return jnp.sqrt(jnp.asarray(el.PI, dtype=rdt) / (two * z)) * jnp.exp(-z)


def real_bessel_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    use_asym = (jnp.abs(z) > jnp.asarray(12.0, dtype=rdt)) & (z > jnp.asarray(0.0, dtype=rdt))
    return jnp.where(use_asym, real_bessel_asym_j(nu, z), real_bessel_series(nu, z, -1.0))


def real_bessel_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    use_asym = (jnp.abs(z) > jnp.asarray(12.0, dtype=rdt)) & (z > jnp.asarray(0.0, dtype=rdt))
    return jnp.where(use_asym, real_bessel_asym_i(nu, z), real_bessel_series(nu, z, 1.0))


def real_bessel_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    s = jnp.sin(jnp.asarray(el.PI, dtype=rdt) * nu)
    jnu = real_bessel_series(nu, z, -1.0)
    jneg = real_bessel_series(-nu, z, -1.0)
    val = (jnu * jnp.cos(jnp.asarray(el.PI, dtype=rdt) * nu) - jneg) / s
    return jnp.where(jnp.abs(s) < jnp.asarray(1e-8, dtype=rdt), jnp.asarray(jnp.inf, dtype=rdt), val)


def real_bessel_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    s = jnp.sin(jnp.asarray(el.PI, dtype=rdt) * nu)
    inu = real_bessel_series(nu, z, 1.0)
    ineg = real_bessel_series(-nu, z, 1.0)
    val = jnp.asarray(el.HALF_PI, dtype=rdt) * (ineg - inu) / s
    return jnp.where(jnp.abs(s) < jnp.asarray(1e-8, dtype=rdt), jnp.asarray(jnp.inf, dtype=rdt), val)


def real_bessel_eval_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    return real_bessel_j(nu, z)


def real_bessel_eval_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    return real_bessel_i(nu, z)


def real_bessel_eval_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    use_asym = (jnp.abs(z) > jnp.asarray(12.0, dtype=rdt)) & (z > jnp.asarray(0.0, dtype=rdt))
    return jnp.where(use_asym, real_bessel_asym_y(nu, z), real_bessel_y(nu, z))


def real_bessel_eval_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    rdt = _real_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=rdt)
    z = jnp.asarray(z, dtype=rdt)
    use_asym = (jnp.abs(z) > jnp.asarray(12.0, dtype=rdt)) & (z > jnp.asarray(0.0, dtype=rdt))
    return jnp.where(use_asym, real_bessel_asym_k(nu, z), real_bessel_k(nu, z))


def complex_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    cdt = _complex_dtype(nu, z)
    rdt = el.real_dtype_from_complex_dtype(cdt)
    nu = jnp.asarray(nu, dtype=cdt)
    z = jnp.asarray(z, dtype=cdt)
    sign = jnp.asarray(sign, dtype=rdt)
    half = jnp.asarray(0.5, dtype=rdt) * z
    quarter = jnp.asarray(0.25, dtype=rdt)
    pow_half = jnp.exp(nu * jnp.log(half))
    gamma = jnp.exp(barnesg._complex_loggamma(nu + jnp.asarray(1.0 + 0.0j, dtype=cdt)))
    term0 = pow_half / gamma
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.asarray(k + 1, dtype=rdt)
        den = k1 * (nu + k1)
        num = (quarter * sign) * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, _BESSEL_COMPLEX_TERMS - 1, body, (term0, sum0))
    return s


def complex_bessel_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    cdt = _complex_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=cdt)
    z = jnp.asarray(z, dtype=cdt)
    s = jnp.sin(jnp.asarray(el.PI, dtype=cdt) * nu)
    jnu = complex_bessel_series(nu, z, -1.0)
    jneg = complex_bessel_series(-nu, z, -1.0)
    val = (jnu * jnp.cos(jnp.asarray(el.PI, dtype=cdt) * nu) - jneg) / s
    return jnp.where(jnp.abs(s) < jnp.asarray(1e-8, dtype=el.real_dtype_from_complex_dtype(cdt)), jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=cdt), val)


def complex_bessel_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    cdt = _complex_dtype(nu, z)
    nu = jnp.asarray(nu, dtype=cdt)
    z = jnp.asarray(z, dtype=cdt)
    s = jnp.sin(jnp.asarray(el.PI, dtype=cdt) * nu)
    inu = complex_bessel_series(nu, z, 1.0)
    ineg = complex_bessel_series(-nu, z, 1.0)
    val = jnp.asarray(el.HALF_PI, dtype=cdt) * (ineg - inu) / s
    return jnp.where(jnp.abs(s) < jnp.asarray(1e-8, dtype=el.real_dtype_from_complex_dtype(cdt)), jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=cdt), val)


__all__ = [
    "real_bessel_series",
    "real_bessel_asym_j",
    "real_bessel_asym_y",
    "real_bessel_asym_i",
    "real_bessel_asym_k",
    "real_bessel_j",
    "real_bessel_i",
    "real_bessel_y",
    "real_bessel_k",
    "real_bessel_eval_j",
    "real_bessel_eval_i",
    "real_bessel_eval_y",
    "real_bessel_eval_k",
    "complex_bessel_series",
    "complex_bessel_y",
    "complex_bessel_k",
]
