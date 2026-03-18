from __future__ import annotations

import jax
import jax.numpy as jnp

from . import jax_precision


# Central constants
PI = jnp.float64(3.1415926535897932384626433832795028841971)
TWO_PI = jnp.float64(2.0) * PI
HALF_PI = jnp.float64(0.5) * PI
LOG_PI = jnp.log(PI)
SQRT_PI = jnp.sqrt(PI)
LOG_TWO = jnp.log(jnp.float64(2.0))
LOG_TWO_PI = jnp.log(TWO_PI)
LOG_SQRT_TWO_PI = jnp.float64(0.5) * LOG_TWO_PI
EULER_GAMMA = jnp.float64(0.577215664901532860606512090082402431)
TWO_OVER_SQRT_PI = jnp.float64(2.0) / SQRT_PI
SQRT_TWO_OVER_PI = jnp.sqrt(jnp.float64(2.0) / PI)
SQRT_PI_OVER_TWO = jnp.sqrt(PI / jnp.float64(2.0))
E = jnp.exp(jnp.float64(1.0))
I = jnp.complex128(1j)


def _dtype_from(*xs: jax.Array):
    if not xs:
        return jnp.float64
    return jnp.result_type(*[jnp.asarray(x).dtype for x in xs])


def pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(PI, dtype=_dtype_from(*xs))


def two_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(TWO_PI, dtype=_dtype_from(*xs))


def half_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(HALF_PI, dtype=_dtype_from(*xs))


def log_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(LOG_PI, dtype=_dtype_from(*xs))


def sqrt_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(SQRT_PI, dtype=_dtype_from(*xs))


def log_two_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(LOG_TWO, dtype=_dtype_from(*xs))


def log_two_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(LOG_TWO_PI, dtype=_dtype_from(*xs))


def log_sqrt_two_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(LOG_SQRT_TWO_PI, dtype=_dtype_from(*xs))


def euler_gamma_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(EULER_GAMMA, dtype=_dtype_from(*xs))


def two_over_sqrt_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(TWO_OVER_SQRT_PI, dtype=_dtype_from(*xs))


def sqrt_two_over_pi_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(SQRT_TWO_OVER_PI, dtype=_dtype_from(*xs))


def sqrt_pi_over_two_like(*xs: jax.Array) -> jax.Array:
    return jnp.asarray(SQRT_PI_OVER_TWO, dtype=_dtype_from(*xs))


def as_real(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.floating):
        return xx
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return jnp.asarray(jnp.real(xx), dtype=jnp.float32 if xx.dtype == jnp.complex64 else jnp.float64)
    return jnp.asarray(xx, dtype=jnp.float64)


def as_complex(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return xx
    if jnp.issubdtype(xx.dtype, jnp.floating):
        return xx.astype(jnp.complex64 if xx.dtype == jnp.float32 else jnp.complex128)
    return jnp.asarray(xx, dtype=jnp.complex128)


def promote_dtype(*xs: jax.Array):
    dt = jnp.result_type(*xs)
    return tuple(jnp.asarray(x, dtype=dt) for x in xs)


def complex_promote(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return xx
    if jnp.issubdtype(xx.dtype, jnp.floating):
        return xx.astype(jnp.complex64 if xx.dtype == jnp.float32 else jnp.complex128)
    return xx.astype(jnp.complex128)


def complex_dtype_from(*xs: jax.Array) -> jnp.dtype:
    if not xs:
        return jnp.dtype(jnp.complex128)
    dt = jnp.result_type(*[jnp.asarray(x).dtype for x in xs])
    if jnp.issubdtype(dt, jnp.complexfloating):
        return jnp.dtype(jnp.complex64) if dt == jnp.dtype(jnp.complex64) else jnp.dtype(jnp.complex128)
    return jnp.dtype(jnp.complex64) if dt == jnp.dtype(jnp.float32) else jnp.dtype(jnp.complex128)


def real_dtype_from_complex_dtype(dt: jnp.dtype) -> jnp.dtype:
    return jnp.dtype(jnp.float32) if jnp.dtype(dt) == jnp.dtype(jnp.complex64) else jnp.dtype(jnp.float64)


def eps(dtype=jnp.float64) -> jax.Array:
    return jnp.finfo(dtype).eps


def tiny(dtype=jnp.float64) -> jax.Array:
    return jnp.finfo(dtype).tiny


def max_value(dtype=jnp.float64) -> jax.Array:
    return jnp.finfo(dtype).max


def safe_div(a: jax.Array, b: jax.Array, fill: float = 0.0) -> jax.Array:
    aa, bb = promote_dtype(a, b)
    denom_zero = bb == 0
    safe_b = jnp.where(denom_zero, jnp.asarray(1, dtype=aa.dtype), bb)
    out = aa / safe_b
    return jnp.where(denom_zero, jnp.asarray(fill, dtype=aa.dtype), out)


def logaddexp(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.logaddexp(a, b)


def logsubexp(a: jax.Array, b: jax.Array) -> jax.Array:
    aa, bb = promote_dtype(a, b)
    return aa + jnp.log1p(-jnp.exp(bb - aa))


def logsumexp(v: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
    return jax_precision.safe_logsumexp(v, axis=axis, keepdims=keepdims)


def log1mexp(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    cutoff = -log_two_like(xx)
    return jnp.where(xx < cutoff, jnp.log1p(-jnp.exp(xx)), jnp.log(-jnp.expm1(xx)))


def logexpm1(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    cutoff = log_two_like(xx)
    return jnp.where(xx > cutoff, xx + jnp.log1p(-jnp.exp(-xx)), jnp.log(jnp.expm1(xx)))


def log_abs(z: jax.Array) -> jax.Array:
    zz = complex_promote(z)
    return jnp.log(jnp.abs(zz))


def log_pow_abs(x: jax.Array, a: jax.Array) -> jax.Array:
    aa, xx = promote_dtype(a, jnp.asarray(x))
    return aa * jnp.log(jnp.abs(xx))


def x_pow_a(x: jax.Array, a: jax.Array) -> jax.Array:
    xx = complex_promote(x)
    aa = complex_promote(a)
    return jnp.exp(aa * jnp.log(xx))


def cis(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.cos(xx) + 1j * jnp.sin(xx)


def sinc(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.where(jnp.abs(xx) < 1e-15, jnp.asarray(1, dtype=xx.dtype), jnp.sin(xx) / xx)


def sinc_pi(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    t = pi_like(xx) * xx
    return jnp.where(jnp.abs(xx) < 1e-15, jnp.asarray(1, dtype=xx.dtype), jnp.sin(t) / t)


def sin_pi(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.sin(pi_like(xx) * xx)


def cos_pi(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.cos(pi_like(xx) * xx)


def tan_pi(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.tan(pi_like(xx) * xx)


def exp_pi_i(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.exp(1j * pi_like(xx) * xx)


def log_sin_pi(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return jnp.log(sin_pi(xx))


def clog(z: jax.Array) -> jax.Array:
    zz = complex_promote(z)
    return jnp.log(jnp.abs(zz)) + 1j * jnp.angle(zz)


def cpow(z: jax.Array, a: jax.Array) -> jax.Array:
    zz = complex_promote(z)
    aa = complex_promote(a)
    return jnp.exp(aa * clog(zz))


def z_to_minus_s(z: jax.Array, s: jax.Array) -> jax.Array:
    zz = complex_promote(z)
    ss = complex_promote(s)
    return jnp.exp(-ss * clog(zz))


__all__ = [
    "PI",
    "TWO_PI",
    "HALF_PI",
    "LOG_PI",
    "SQRT_PI",
    "LOG_TWO",
    "LOG_TWO_PI",
    "LOG_SQRT_TWO_PI",
    "EULER_GAMMA",
    "TWO_OVER_SQRT_PI",
    "SQRT_TWO_OVER_PI",
    "SQRT_PI_OVER_TWO",
    "E",
    "I",
    "pi_like",
    "two_pi_like",
    "half_pi_like",
    "log_pi_like",
    "sqrt_pi_like",
    "log_two_like",
    "log_two_pi_like",
    "log_sqrt_two_pi_like",
    "euler_gamma_like",
    "two_over_sqrt_pi_like",
    "sqrt_two_over_pi_like",
    "sqrt_pi_over_two_like",
    "as_real",
    "as_complex",
    "promote_dtype",
    "complex_promote",
    "complex_dtype_from",
    "real_dtype_from_complex_dtype",
    "eps",
    "tiny",
    "max_value",
    "safe_div",
    "logaddexp",
    "logsubexp",
    "logsumexp",
    "log1mexp",
    "logexpm1",
    "log_abs",
    "log_pow_abs",
    "x_pow_a",
    "cis",
    "sinc",
    "sinc_pi",
    "sin_pi",
    "cos_pi",
    "tan_pi",
    "exp_pi_i",
    "log_sin_pi",
    "clog",
    "cpow",
    "z_to_minus_s",
]
