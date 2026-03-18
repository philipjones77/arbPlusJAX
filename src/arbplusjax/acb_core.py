from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import double_interval as di
from . import arb_core
from . import acb_dirichlet
from . import barnesg
from . import elementary as el
from . import hypgeom
from . import checks



def as_acb_box(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    checks.check_last_dim(arr, 4, "acb_core.as_acb_box")
    return arr


def _real_dtype_for_box(x: jax.Array) -> jnp.dtype:
    return jnp.asarray(x).dtype


def _complex_dtype_for_box(x: jax.Array) -> jnp.dtype:
    return jnp.dtype(jnp.complex64) if _real_dtype_for_box(x) == jnp.dtype(jnp.float32) else jnp.dtype(jnp.complex128)


def _real_scalar_for_box(x: jax.Array, value: float) -> jax.Array:
    return jnp.asarray(value, dtype=_real_dtype_for_box(x))


def _complex_scalar_for_box(x: jax.Array, value: complex) -> jax.Array:
    return jnp.asarray(value, dtype=_complex_dtype_for_box(x))


def acb_box(real_interval: jax.Array, imag_interval: jax.Array) -> jax.Array:
    r = di.as_interval(real_interval)
    i = di.as_interval(imag_interval)
    return jnp.concatenate([r, i], axis=-1)


def acb_real(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return box[..., 0:2]


def acb_imag(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return box[..., 2:4]


def acb_midpoint(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = di.midpoint(acb_real(box))
    im = di.midpoint(acb_imag(box))
    return re + 1j * im


def acb_zero() -> jax.Array:
    z = jnp.asarray(0.0, dtype=jnp.float64)
    return acb_box(di.interval(z, z), di.interval(z, z))


def acb_one() -> jax.Array:
    o = jnp.asarray(1.0, dtype=jnp.float64)
    z = jnp.asarray(0.0, dtype=jnp.float64)
    return acb_box(di.interval(o, o), di.interval(z, z))


def acb_onei() -> jax.Array:
    o = jnp.asarray(1.0, dtype=jnp.float64)
    z = jnp.asarray(0.0, dtype=jnp.float64)
    return acb_box(di.interval(z, z), di.interval(o, o))


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=x.dtype)
    inf = jnp.inf * t
    return acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def _acb_from_complex(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_box(di.interval(di._below(re), di._above(re)), di.interval(di._below(im), di._above(im)))


def _acb_unary_from_midpoint(x: jax.Array, fn) -> jax.Array:
    box = as_acb_box(x)
    z = acb_midpoint(box)
    v = fn(z)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(box))


def _acb_unary_pair_from_midpoint(x: jax.Array, fn) -> tuple[jax.Array, jax.Array]:
    box = as_acb_box(x)
    z = acb_midpoint(box)
    v1, v2 = fn(z)
    finite1 = jnp.isfinite(jnp.real(v1)) & jnp.isfinite(jnp.imag(v1))
    finite2 = jnp.isfinite(jnp.real(v2)) & jnp.isfinite(jnp.imag(v2))
    out1 = _acb_from_complex(v1)
    out2 = _acb_from_complex(v2)
    full = _full_box_like(box)
    out1 = jnp.where(finite1[..., None], out1, full)
    out2 = jnp.where(finite2[..., None], out2, full)
    return out1, out2


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    z = el.as_complex(z)
    complex_dtype = z.dtype
    real_dtype = jnp.real(z).dtype
    z1 = z - jnp.asarray(1.0 + 0.0j, dtype=complex_dtype)
    coeffs = jnp.asarray(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ],
        dtype=complex_dtype,
    )
    x = coeffs[0]
    for k in range(1, coeffs.shape[0]):
        x = x + coeffs[k] / (z1 + jnp.asarray(k, dtype=real_dtype))
    t = z1 + jnp.asarray(7.5, dtype=real_dtype)
    return jnp.asarray(el.LOG_SQRT_TWO_PI, dtype=complex_dtype) + (z1 + jnp.asarray(0.5, dtype=real_dtype)) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = el.as_complex(z)
    complex_dtype = z.dtype

    def reflection(w):
        return jnp.asarray(el.LOG_PI, dtype=complex_dtype) - jnp.log(jnp.sin(jnp.asarray(el.PI, dtype=complex_dtype) * w)) - _complex_loggamma_lanczos(jnp.asarray(1.0, dtype=complex_dtype) - w)

    return lax.cond(jnp.real(z) < 0.5, reflection, _complex_loggamma_lanczos, z)


def acb_is_exact(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    return (re[..., 0] == re[..., 1]) & (im[..., 0] == im[..., 1])


def acb_is_zero(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    return (re[..., 0] == 0.0) & (re[..., 1] == 0.0) & (im[..., 0] == 0.0) & (im[..., 1] == 0.0)


def acb_is_one(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    return (re[..., 0] == 1.0) & (re[..., 1] == 1.0) & (im[..., 0] == 0.0) & (im[..., 1] == 0.0)


def acb_is_real(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    im = acb_imag(box)
    return (im[..., 0] == 0.0) & (im[..., 1] == 0.0)


def acb_is_int(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    exact = (re[..., 0] == re[..., 1]) & (im[..., 0] == 0.0) & (im[..., 1] == 0.0)
    return exact & (re[..., 0] == jnp.floor(re[..., 0]))


def acb_set(x: jax.Array) -> jax.Array:
    return as_acb_box(x)


def acb_neg(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return acb_box(di.neg(acb_real(box)), di.neg(acb_imag(box)))


def acb_conj(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return acb_box(acb_real(box), di.neg(acb_imag(box)))


def acb_add(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    return acb_box(di.fast_add(acb_real(xb), acb_real(yb)), di.fast_add(acb_imag(xb), acb_imag(yb)))


def acb_sub(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    return acb_box(di.fast_sub(acb_real(xb), acb_real(yb)), di.fast_sub(acb_imag(xb), acb_imag(yb)))


def acb_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    xr, xi = acb_real(xb), acb_imag(xb)
    yr, yi = acb_real(yb), acb_imag(yb)
    ac = di.fast_mul(xr, yr)
    bd = di.fast_mul(xi, yi)
    ad = di.fast_mul(xr, yi)
    bc = di.fast_mul(xi, yr)
    return acb_box(di.fast_sub(ac, bd), di.fast_add(ad, bc))


def acb_mul_naive(x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_mul(x, y)


def acb_mul_onei(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return acb_box(di.neg(acb_imag(box)), acb_real(box))


def acb_div_onei(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    return acb_box(acb_imag(box), di.neg(acb_real(box)))


def acb_inv(x: jax.Array) -> jax.Array:
    return acb_div(acb_one(), x)


def acb_div(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    yr, yi = acb_real(yb), acb_imag(yb)
    den = di.fast_add(di.fast_mul(yr, yr), di.fast_mul(yi, yi))
    num = acb_box(yr, di.neg(yi))
    out = acb_mul(xb, num)
    return acb_box(di.fast_div(acb_real(out), den), di.fast_div(acb_imag(out), den))


def acb_abs(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    r2 = di.fast_add(di.fast_mul(re, re), di.fast_mul(im, im))
    return arb_core.arb_sqrt(r2)


def acb_arg(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.arctan2(jnp.imag(m), jnp.real(m))
    finite = jnp.isfinite(val)
    out = di.interval(di._below(val), di._above(val))
    full = di.interval(-jnp.inf, jnp.inf)
    return jnp.where(finite, out, full)


def acb_sgn(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    mag = jnp.abs(m)
    v = jnp.where(mag == 0.0, 0.0 + 0.0j, m / mag)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(box))


def acb_csgn(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    re = jnp.real(m)
    im = jnp.imag(m)
    val = jnp.where(re > 0.0, 1.0, jnp.where(re < 0.0, -1.0, jnp.where(im >= 0.0, 1.0, -1.0)))
    return di.interval(di._below(val), di._above(val))


def acb_real_abs(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.abs(jnp.real(m))
    finite = jnp.isfinite(val)
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return jnp.where(finite[..., None], out, _full_box_like(box))


def acb_real_sgn(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.sign(jnp.real(m))
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return out


def acb_real_heaviside(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.where(jnp.real(m) < 0.0, 0.0, 1.0)
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return out


def acb_real_floor(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.floor(jnp.real(m))
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return out


def acb_real_ceil(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = acb_midpoint(box)
    val = jnp.ceil(jnp.real(m))
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return out


def acb_real_max(x: jax.Array, y: jax.Array, analytic: int = 0) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    mx = jnp.maximum(jnp.real(acb_midpoint(xb)), jnp.real(acb_midpoint(yb)))
    out = acb_box(di.interval(di._below(mx), di._above(mx)), di.interval(0.0, 0.0))
    return out


def acb_real_min(x: jax.Array, y: jax.Array, analytic: int = 0) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    mx = jnp.minimum(jnp.real(acb_midpoint(xb)), jnp.real(acb_midpoint(yb)))
    out = acb_box(di.interval(di._below(mx), di._above(mx)), di.interval(0.0, 0.0))
    return out


def acb_real_sqrtpos(x: jax.Array, analytic: int = 0) -> jax.Array:
    box = as_acb_box(x)
    m = jnp.maximum(jnp.real(acb_midpoint(box)), 0.0)
    val = jnp.sqrt(m)
    out = acb_box(di.interval(di._below(val), di._above(val)), di.interval(0.0, 0.0))
    return out


def acb_sqrt_analytic(x: jax.Array, analytic: int = 0) -> jax.Array:
    return acb_sqrt(x)


def acb_rsqrt_analytic(x: jax.Array, analytic: int = 0) -> jax.Array:
    return acb_rsqrt(x)


def acb_log_analytic(x: jax.Array, analytic: int = 0) -> jax.Array:
    return acb_log(x)


def acb_pow_analytic(x: jax.Array, y: jax.Array, analytic: int = 0) -> jax.Array:
    return acb_pow(x, y)


def acb_dot_simple(
    initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array
) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    init = as_acb_box(initial)
    prod = acb_mul(xb, yb)
    total = acb_add(init, jnp.sum(prod, axis=0))
    return acb_sub(init, jnp.sum(prod, axis=0)) if subtract else total


def acb_dot_precise(
    initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array
) -> jax.Array:
    return acb_dot_simple(initial, subtract, x, y)


def acb_dot(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_simple(initial, subtract, x, y)


def acb_approx_dot(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_simple(initial, subtract, x, y)


def acb_dot_ui(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yy = jnp.asarray(y, dtype=xb.dtype)
    zz = jnp.zeros_like(yy)
    yb = acb_box(di.interval(yy, yy), di.interval(zz, zz))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_dot_si(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yy = jnp.asarray(y, dtype=xb.dtype)
    zz = jnp.zeros_like(yy)
    yb = acb_box(di.interval(yy, yy), di.interval(zz, zz))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_dot_uiui(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_ui(initial, subtract, x, y)


def acb_dot_siui(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_ui(initial, subtract, x, y)


def acb_dot_fmpz(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yy = jnp.asarray(y, dtype=xb.dtype)
    zz = jnp.zeros_like(yy)
    yb = acb_box(di.interval(yy, yy), di.interval(zz, zz))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_add_error_arb(x: jax.Array, err: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    er = di.as_interval(err)
    return acb_box(di.fast_add(acb_real(xb), er), di.fast_add(acb_imag(xb), er))


def acb_add_error_arf(x: jax.Array, err: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    e = jnp.asarray(err, dtype=xb.dtype)
    er = di.interval(e, e)
    return acb_add_error_arb(x, er)


def acb_add_error_mag(x: jax.Array, err: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    e = jnp.asarray(err, dtype=xb.dtype)
    er = di.interval(e, e)
    return acb_add_error_arb(x, er)


def acb_union(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    re = di.interval(jnp.minimum(acb_real(xb)[..., 0], acb_real(yb)[..., 0]),
                     jnp.maximum(acb_real(xb)[..., 1], acb_real(yb)[..., 1]))
    im = di.interval(jnp.minimum(acb_imag(xb)[..., 0], acb_imag(yb)[..., 0]),
                     jnp.maximum(acb_imag(xb)[..., 1], acb_imag(yb)[..., 1]))
    return acb_box(re, im)


def acb_trim(x: jax.Array) -> jax.Array:
    return as_acb_box(x)


@jax.jit
def acb_exp(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.exp)


@jax.jit
def acb_log(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.log)


@jax.jit
def acb_log1p(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.log1p)


@jax.jit
def acb_sqrt(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.sqrt)


@jax.jit
def acb_rsqrt(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.sqrt(z))


@jax.jit
def acb_sin(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.sin)


@jax.jit
def acb_cos(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.cos)


@jax.jit
def acb_sin_cos(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _acb_unary_pair_from_midpoint(x, lambda z: (jnp.sin(z), jnp.cos(z)))


@jax.jit
def acb_tan(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.tan)


@jax.jit
def acb_cot(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.tan(z))


@jax.jit
def acb_sinh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.sinh)


@jax.jit
def acb_cosh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.cosh)


@jax.jit
def acb_tanh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.tanh)


@jax.jit
def acb_asin(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arcsin)


@jax.jit
def acb_acos(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arccos)


@jax.jit
def acb_atan(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arctan)


@jax.jit
def acb_asinh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arcsinh)


@jax.jit
def acb_acosh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arccosh)


@jax.jit
def acb_atanh(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.arctanh)


@jax.jit
def acb_sech(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.cosh(z))


@jax.jit
def acb_csch(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.sinh(z))


@jax.jit
def acb_sin_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.sin(el.PI * z))


@jax.jit
def acb_cos_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.cos(el.PI * z))


@jax.jit
def acb_sin_cos_pi(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _acb_unary_pair_from_midpoint(x, lambda z: (jnp.sin(el.PI * z), jnp.cos(el.PI * z)))


@jax.jit
def acb_tan_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.tan(el.PI * z))


@jax.jit
def acb_cot_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.tan(el.PI * z))


@jax.jit
def acb_csc_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.sin(el.PI * z))


@jax.jit
def acb_sinc(x: jax.Array) -> jax.Array:
    def fn(z):
        return jnp.where(jnp.abs(z) < 1e-12, 1.0 + 0.0j, jnp.sin(z) / z)

    return _acb_unary_from_midpoint(x, fn)


@jax.jit
def acb_sinc_pi(x: jax.Array) -> jax.Array:
    def fn(z):
        t = el.PI * z
        return jnp.where(jnp.abs(z) < 1e-12, 1.0 + 0.0j, jnp.sin(t) / t)

    return _acb_unary_from_midpoint(x, fn)


@jax.jit
def acb_exp_pi_i(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.exp(1j * el.PI * z))


@jax.jit
def acb_expm1(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, jnp.expm1)


@jax.jit
def acb_exp_invexp(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _acb_unary_pair_from_midpoint(x, lambda z: (jnp.exp(z), jnp.exp(-z)))


def acb_addmul(z: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_add(as_acb_box(z), acb_mul(x, y))


def acb_submul(z: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_sub(as_acb_box(z), acb_mul(x, y))


@jax.jit
def acb_pow(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    z = acb_midpoint(xb)
    w = acb_midpoint(yb)
    v = el.cpow(z, w)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(xb))


@jax.jit
def acb_pow_arb(x: jax.Array, y: jax.Array) -> jax.Array:
    yb = di.as_interval(y)
    m = di.midpoint(yb)
    zero = jnp.zeros_like(m)
    return acb_pow(x, acb_box(di.interval(m, m), di.interval(zero, zero)))


@jax.jit
def acb_pow_ui(x: jax.Array, n: int) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.power(z, n))


@jax.jit
def acb_pow_si(x: jax.Array, n: int) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.power(z, n))


@jax.jit
def acb_pow_fmpz(x: jax.Array, n: jax.Array) -> jax.Array:
    n_val = jnp.asarray(n, dtype=jnp.int64)
    return _acb_unary_from_midpoint(x, lambda z: jnp.power(z, n_val))


@jax.jit
def acb_sqr(x: jax.Array) -> jax.Array:
    return acb_mul(x, x)


@jax.jit
def acb_root_ui(x: jax.Array, k: int) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.power(z, jnp.asarray(1.0, dtype=jnp.real(z).dtype) / jnp.asarray(k, dtype=jnp.real(z).dtype)))


@jax.jit
def acb_gamma(x: jax.Array) -> jax.Array:
    return hypgeom.acb_hypgeom_gamma(as_acb_box(x))


@jax.jit
def acb_rgamma(x: jax.Array) -> jax.Array:
    return hypgeom.acb_hypgeom_rgamma(as_acb_box(x))


@jax.jit
def acb_lgamma(x: jax.Array) -> jax.Array:
    return hypgeom.acb_hypgeom_lgamma(as_acb_box(x))


@jax.jit
def acb_log_sin_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.log(jnp.sin(el.PI * z)))


@jax.jit
def acb_digamma(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    z = acb_midpoint(box)
    h = _complex_scalar_for_box(box, 1e-6 + 0.0j)
    two = _complex_scalar_for_box(box, 2.0 + 0.0j)
    v = (_complex_loggamma(z + h) - _complex_loggamma(z - h)) / (two * h)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(box))


@jax.jit
def acb_barnes_g(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]
    cross_pole = (im_lo <= 0.0) & (im_hi >= 0.0) & ((jnp.ceil(-re_hi) <= jnp.floor(-re_lo)) & (jnp.floor(-re_lo) >= 0.0))
    half = jnp.asarray(0.5, dtype=re.dtype)
    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            half * (re_lo + re_hi) + 1j * half * (im_lo + im_hi),
        ],
        dtype=_complex_dtype_for_box(box),
    )
    vals = jax.vmap(barnesg.barnesg_complex)(corners)
    finite = jnp.all(jnp.isfinite(jnp.real(vals))) & jnp.all(jnp.isfinite(jnp.imag(vals)))
    out = acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )
    return jnp.where(cross_pole | (~finite), _full_box_like(box), out)


@jax.jit
def acb_log_barnes_g(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    re = acb_real(box)
    im = acb_imag(box)
    re_lo, re_hi = re[0], re[1]
    im_lo, im_hi = im[0], im[1]
    cross_pole = (im_lo <= 0.0) & (im_hi >= 0.0) & ((jnp.ceil(-re_hi) <= jnp.floor(-re_lo)) & (jnp.floor(-re_lo) >= 0.0))
    half = jnp.asarray(0.5, dtype=re.dtype)
    corners = jnp.asarray(
        [
            re_lo + 1j * im_lo,
            re_lo + 1j * im_hi,
            re_hi + 1j * im_lo,
            re_hi + 1j * im_hi,
            half * (re_lo + re_hi) + 1j * half * (im_lo + im_hi),
        ],
        dtype=_complex_dtype_for_box(box),
    )
    vals = jax.vmap(barnesg.log_barnesg)(corners)
    finite = jnp.all(jnp.isfinite(jnp.real(vals))) & jnp.all(jnp.isfinite(jnp.imag(vals)))
    out = acb_box(
        di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals)))),
        di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals)))),
    )
    return jnp.where(cross_pole | (~finite), _full_box_like(box), out)


@jax.jit
def acb_zeta(x: jax.Array) -> jax.Array:
    return acb_dirichlet.acb_dirichlet_zeta(as_acb_box(x))


@partial(jax.jit, static_argnames=("terms", "max_terms", "min_terms"))
def acb_hurwitz_zeta(
    s: jax.Array,
    a: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    sb = as_acb_box(s)
    ab = as_acb_box(a)
    zs = acb_midpoint(sb)
    za = acb_midpoint(ab)
    re_s = jnp.real(zs)
    eps = jnp.float64(1e-12)
    tail_target = eps * jnp.maximum(re_s - 1.0, 1e-12)
    base = jnp.power(tail_target, 1.0 / jnp.maximum(1.0 - re_s, 1e-12))
    n_est = jnp.ceil(base + 1.0)
    n_eff = jnp.where(re_s > 1.1, n_est, jnp.float64(terms))
    n_eff = jnp.clip(n_eff, jnp.float64(min_terms), jnp.float64(max_terms))

    ks = jnp.arange(max_terms, dtype=jnp.float64)
    mask = ks < n_eff
    v = jnp.sum(jnp.where(mask, jnp.power(za + ks, -zs), 0.0 + 0.0j))
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(sb))


@partial(jax.jit, static_argnames=("m", "terms", "max_terms", "min_terms"))
def acb_polygamma(
    m: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    if m == 0:
        return acb_digamma(z)
    zb = as_acb_box(z)
    zmid = acb_midpoint(zb)
    re_z = jnp.real(zmid)
    m_float = jnp.float64(m)
    eps = jnp.float64(1e-12)
    tail_target = eps * jnp.maximum(m_float, 1.0)
    base = jnp.power(tail_target, -1.0 / jnp.maximum(m_float, 1e-12))
    n_est = jnp.ceil(base - re_z)
    n_eff = jnp.where(m_float > 0, n_est, jnp.float64(terms))
    n_eff = jnp.clip(n_eff, jnp.float64(min_terms), jnp.float64(max_terms))

    ks = jnp.arange(max_terms, dtype=jnp.float64)
    mask = ks < n_eff
    factorial = jnp.exp(lax.lgamma(m_float + 1.0))
    series = jnp.sum(jnp.where(mask, jnp.power(zmid + ks, -(m_float + 1.0)), 0.0 + 0.0j))
    v = ((-1.0) ** (m + 1)) * factorial * series
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(zb))


@partial(jax.jit, static_argnames=("n",))
def acb_bernoulli_poly_ui(n: int, x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    z = acb_midpoint(xb)

    def fallback():
        return _full_box_like(xb)

    complex_dtype = _complex_dtype_for_box(xb)
    real_dtype = _real_dtype_for_box(xb)
    if n == 0:
        v = jnp.asarray(1.0 + 0.0j, dtype=complex_dtype)
    elif n == 1:
        v = z - jnp.asarray(0.5, dtype=real_dtype)
    elif n == 2:
        v = z * z - z + jnp.asarray(1.0 / 6.0, dtype=real_dtype)
    elif n == 3:
        v = z * z * z - jnp.asarray(1.5, dtype=real_dtype) * z * z + jnp.asarray(0.5, dtype=real_dtype) * z
    elif n == 4:
        v = z**4 - jnp.asarray(2.0, dtype=real_dtype) * z**3 + z * z - jnp.asarray(1.0 / 30.0, dtype=real_dtype)
    else:
        return fallback()

    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(xb))


@partial(jax.jit, static_argnames=("terms", "max_terms", "min_terms"))
def acb_polylog(
    s: jax.Array,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    sb = as_acb_box(s)
    zb = as_acb_box(z)
    smid = acb_midpoint(sb)
    zmid = acb_midpoint(zb)
    absz = jnp.abs(zmid)
    eps = jnp.float64(1e-12)
    base = jnp.ceil(8.0 / jnp.maximum(1.0 - absz, eps))
    n_eff = jnp.where(absz < 1.0, base, jnp.float64(terms))
    n_eff = jnp.clip(n_eff, jnp.float64(min_terms), jnp.float64(max_terms))

    ks = jnp.arange(1, max_terms + 1, dtype=jnp.float64)
    mask = ks <= n_eff
    v = jnp.sum(jnp.where(mask, jnp.power(zmid, ks) / jnp.power(ks, smid), 0.0 + 0.0j))
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v)) & (absz < 1.0)
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(zb))


@partial(jax.jit, static_argnames=("s", "terms", "max_terms", "min_terms"))
def acb_polylog_si(
    s: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
) -> jax.Array:
    zbox = as_acb_box(z)
    zero = jnp.asarray(0.0, dtype=zbox.dtype)
    sval = jnp.asarray(s, dtype=zbox.dtype)
    s_box = acb_box(
        di.interval(sval, sval),
        di.interval(zero, zero),
    )
    return acb_polylog(s_box, z, terms=terms, max_terms=max_terms, min_terms=min_terms)


@partial(jax.jit, static_argnames=("iters",))
def acb_agm(a: jax.Array, b: jax.Array, iters: int = 10) -> jax.Array:
    abox = as_acb_box(a)
    bbox = as_acb_box(b)
    a0 = acb_midpoint(abox)
    b0 = acb_midpoint(bbox)

    def body(_, state):
        aa, bb = state
        a1 = 0.5 * (aa + bb)
        b1 = jnp.sqrt(aa * bb)
        return a1, b1

    a1, _ = lax.fori_loop(0, iters, body, (a0, b0))
    finite = jnp.isfinite(jnp.real(a1)) & jnp.isfinite(jnp.imag(a1))
    out = _acb_from_complex(a1)
    return jnp.where(finite[..., None], out, _full_box_like(abox))


@partial(jax.jit, static_argnames=("iters",))
def acb_agm1(x: jax.Array, iters: int = 10) -> jax.Array:
    return acb_agm(acb_one(), x, iters=iters)


@partial(jax.jit, static_argnames=("iters",))
def acb_agm1_cpx(x: jax.Array, iters: int = 10) -> jax.Array:
    return acb_agm1(x, iters=iters)


def acb_box_round_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    box = as_acb_box(x)
    return acb_box(
        di.round_interval_outward(acb_real(box), prec_bits),
        di.round_interval_outward(acb_imag(box), prec_bits),
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_exp_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_exp(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_log_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_log(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sqrt_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sqrt(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sin_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sin(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_cos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_cos(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_tan_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_tan(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sinh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sinh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_cosh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_cosh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_tanh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_tanh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_asin_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_asin(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_acos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_acos(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_atan_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_atan(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_asinh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_asinh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_acosh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_acosh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_atanh_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_atanh(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_abs_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(acb_abs(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_add_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_add(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sub_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sub(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mul_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_mul(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_div_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_div(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_inv_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_inv(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_fma_prec(x: jax.Array, y: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_add(acb_mul(x, y), z), prec_bits)


def _acb_round_output(out, prec_bits: int):
    if isinstance(out, tuple):
        return tuple(_acb_round_output(item, prec_bits) for item in out)
    arr = jnp.asarray(out)
    last = arr.shape[-1] if arr.ndim > 0 else None
    if last == 4:
        return acb_box_round_prec(arr, prec_bits)
    if last == 2:
        return di.round_interval_outward(arr, prec_bits)
    return out


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_log1p_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_log1p(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_rsqrt_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_rsqrt(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sin_cos_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS):
    return _acb_round_output(acb_sin_cos(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_cot_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_cot(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sech_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sech(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_csch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_csch(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sin_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sin_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_cos_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_cos_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sin_cos_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS):
    return _acb_round_output(acb_sin_cos_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_tan_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_tan_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_cot_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_cot_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_csc_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_csc_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sinc_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sinc(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sinc_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sinc_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_exp_pi_i_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_exp_pi_i(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_expm1_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_expm1(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_exp_invexp_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS):
    return _acb_round_output(acb_exp_invexp(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_addmul_prec(z: jax.Array, x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_addmul(z, x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_submul_prec(z: jax.Array, x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_submul(z, x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_pow_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_pow(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_pow_arb_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_pow_arb(x, y), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_pow_ui_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_pow_ui(x, n), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_pow_si_prec(x: jax.Array, n: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_pow_si(x, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_pow_fmpz_prec(x: jax.Array, n: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_pow_fmpz(x, n), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_sqr_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sqr(x), prec_bits)


@partial(jax.jit, static_argnames=("k", "prec_bits"))
def acb_root_ui_prec(x: jax.Array, k: int, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_root_ui(x, k), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_gamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_gamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_rgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_rgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_lgamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_lgamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_log_sin_pi_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_log_sin_pi(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_digamma_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_digamma(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_barnes_g_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_barnes_g(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_log_barnes_g_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_log_barnes_g(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_zeta_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_zeta(x), prec_bits)


@partial(jax.jit, static_argnames=("terms", "max_terms", "min_terms", "prec_bits"))
def acb_hurwitz_zeta_prec(
    s: jax.Array,
    a: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _acb_round_output(acb_hurwitz_zeta(s, a, terms, max_terms, min_terms), prec_bits)


@partial(jax.jit, static_argnames=("m", "terms", "max_terms", "min_terms", "prec_bits"))
def acb_polygamma_prec(
    m: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _acb_round_output(acb_polygamma(m, z, terms, max_terms, min_terms), prec_bits)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def acb_bernoulli_poly_ui_prec(n: int, x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_bernoulli_poly_ui(n, x), prec_bits)


@partial(jax.jit, static_argnames=("terms", "max_terms", "min_terms", "prec_bits"))
def acb_polylog_prec(
    s: jax.Array,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _acb_round_output(acb_polylog(s, z, terms, max_terms, min_terms), prec_bits)


@partial(jax.jit, static_argnames=("terms", "max_terms", "min_terms", "prec_bits"))
def acb_polylog_si_prec(
    s: int,
    z: jax.Array,
    terms: int = 64,
    max_terms: int = 512,
    min_terms: int = 32,
    prec_bits: int = di.DEFAULT_PREC_BITS,
) -> jax.Array:
    return _acb_round_output(acb_polylog_si(s, z, terms, max_terms, min_terms), prec_bits)


@partial(jax.jit, static_argnames=("iters", "prec_bits"))
def acb_agm_prec(a: jax.Array, b: jax.Array, iters: int = 10, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_agm(a, b, iters), prec_bits)


@partial(jax.jit, static_argnames=("iters", "prec_bits"))
def acb_agm1_prec(x: jax.Array, iters: int = 10, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_agm1(x, iters), prec_bits)


@partial(jax.jit, static_argnames=("iters", "prec_bits"))
def acb_agm1_cpx_prec(x: jax.Array, iters: int = 10, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_agm1_cpx(x, iters), prec_bits)


def acb_exp_batch(x: jax.Array) -> jax.Array:
    return acb_exp(as_acb_box(x))


def acb_log_batch(x: jax.Array) -> jax.Array:
    return acb_log(as_acb_box(x))


def acb_sqrt_batch(x: jax.Array) -> jax.Array:
    return acb_sqrt(as_acb_box(x))


def acb_sin_batch(x: jax.Array) -> jax.Array:
    return acb_sin(as_acb_box(x))


def acb_cos_batch(x: jax.Array) -> jax.Array:
    return acb_cos(as_acb_box(x))


def acb_tan_batch(x: jax.Array) -> jax.Array:
    return acb_tan(as_acb_box(x))


def acb_sinh_batch(x: jax.Array) -> jax.Array:
    return acb_sinh(as_acb_box(x))


def acb_cosh_batch(x: jax.Array) -> jax.Array:
    return acb_cosh(as_acb_box(x))


def acb_tanh_batch(x: jax.Array) -> jax.Array:
    return acb_tanh(as_acb_box(x))


def acb_asin_batch(x: jax.Array) -> jax.Array:
    return acb_asin(as_acb_box(x))


def acb_acos_batch(x: jax.Array) -> jax.Array:
    return acb_acos(as_acb_box(x))


def acb_atan_batch(x: jax.Array) -> jax.Array:
    return acb_atan(as_acb_box(x))


def acb_asinh_batch(x: jax.Array) -> jax.Array:
    return acb_asinh(as_acb_box(x))


def acb_acosh_batch(x: jax.Array) -> jax.Array:
    return acb_acosh(as_acb_box(x))


def acb_atanh_batch(x: jax.Array) -> jax.Array:
    return acb_atanh(as_acb_box(x))


def acb_abs_batch(x: jax.Array) -> jax.Array:
    return acb_abs(as_acb_box(x))


def acb_add_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_add(as_acb_box(x), as_acb_box(y))


def acb_sub_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_sub(as_acb_box(x), as_acb_box(y))


def acb_mul_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_mul(as_acb_box(x), as_acb_box(y))


def acb_div_batch(x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_div(as_acb_box(x), as_acb_box(y))


def acb_inv_batch(x: jax.Array) -> jax.Array:
    return acb_inv(as_acb_box(x))


def acb_fma_batch(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    return acb_add(acb_mul(as_acb_box(x), as_acb_box(y)), as_acb_box(z))


def acb_log1p_batch(x: jax.Array) -> jax.Array:
    return acb_log1p(as_acb_box(x))


def acb_expm1_batch(x: jax.Array) -> jax.Array:
    return acb_expm1(as_acb_box(x))


def acb_sin_cos_batch(x: jax.Array):
    return acb_sin_cos(as_acb_box(x))


def acb_exp_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_exp_batch(x), prec_bits)


def acb_log_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_log_batch(x), prec_bits)


def acb_sqrt_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sqrt_batch(x), prec_bits)


def acb_sin_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sin_batch(x), prec_bits)


def acb_cos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_cos_batch(x), prec_bits)


def acb_tan_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_tan_batch(x), prec_bits)


def acb_sinh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_sinh_batch(x), prec_bits)


def acb_cosh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_cosh_batch(x), prec_bits)


def acb_tanh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_box_round_prec(acb_tanh_batch(x), prec_bits)


def acb_asin_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_asin_batch(x), prec_bits)


def acb_acos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_acos_batch(x), prec_bits)


def acb_atan_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_atan_batch(x), prec_bits)


def acb_asinh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_asinh_batch(x), prec_bits)


def acb_acosh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_acosh_batch(x), prec_bits)


def acb_atanh_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_atanh_batch(x), prec_bits)


def acb_abs_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return di.round_interval_outward(acb_abs_batch(x), prec_bits)


def acb_add_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_add_batch(x, y), prec_bits)


def acb_sub_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_sub_batch(x, y), prec_bits)


def acb_mul_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_mul_batch(x, y), prec_bits)


def acb_div_batch_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_div_batch(x, y), prec_bits)


def acb_inv_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_inv_batch(x), prec_bits)


def acb_fma_batch_prec(x: jax.Array, y: jax.Array, z: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_fma_batch(x, y, z), prec_bits)


def acb_log1p_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_log1p_batch(x), prec_bits)


def acb_expm1_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _acb_round_output(acb_expm1_batch(x), prec_bits)


def acb_sin_cos_batch_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS):
    return _acb_round_output(acb_sin_cos_batch(x), prec_bits)


acb_exp_batch_jit = jax.jit(acb_exp_batch)
acb_log_batch_jit = jax.jit(acb_log_batch)
acb_sqrt_batch_jit = jax.jit(acb_sqrt_batch)
acb_sin_batch_jit = jax.jit(acb_sin_batch)
acb_cos_batch_jit = jax.jit(acb_cos_batch)
acb_tan_batch_jit = jax.jit(acb_tan_batch)
acb_sinh_batch_jit = jax.jit(acb_sinh_batch)
acb_cosh_batch_jit = jax.jit(acb_cosh_batch)
acb_tanh_batch_jit = jax.jit(acb_tanh_batch)
acb_asin_batch_jit = jax.jit(acb_asin_batch)
acb_acos_batch_jit = jax.jit(acb_acos_batch)
acb_atan_batch_jit = jax.jit(acb_atan_batch)
acb_asinh_batch_jit = jax.jit(acb_asinh_batch)
acb_acosh_batch_jit = jax.jit(acb_acosh_batch)
acb_atanh_batch_jit = jax.jit(acb_atanh_batch)
acb_abs_batch_jit = jax.jit(acb_abs_batch)
acb_add_batch_jit = jax.jit(acb_add_batch)
acb_sub_batch_jit = jax.jit(acb_sub_batch)
acb_mul_batch_jit = jax.jit(acb_mul_batch)
acb_div_batch_jit = jax.jit(acb_div_batch)
acb_inv_batch_jit = jax.jit(acb_inv_batch)
acb_fma_batch_jit = jax.jit(acb_fma_batch)
acb_log1p_batch_jit = jax.jit(acb_log1p_batch)
acb_expm1_batch_jit = jax.jit(acb_expm1_batch)
acb_sin_cos_batch_jit = jax.jit(acb_sin_cos_batch)

acb_exp_batch_prec_jit = jax.jit(acb_exp_batch_prec, static_argnames=("prec_bits",))
acb_log_batch_prec_jit = jax.jit(acb_log_batch_prec, static_argnames=("prec_bits",))
acb_sqrt_batch_prec_jit = jax.jit(acb_sqrt_batch_prec, static_argnames=("prec_bits",))
acb_sin_batch_prec_jit = jax.jit(acb_sin_batch_prec, static_argnames=("prec_bits",))
acb_cos_batch_prec_jit = jax.jit(acb_cos_batch_prec, static_argnames=("prec_bits",))
acb_tan_batch_prec_jit = jax.jit(acb_tan_batch_prec, static_argnames=("prec_bits",))
acb_sinh_batch_prec_jit = jax.jit(acb_sinh_batch_prec, static_argnames=("prec_bits",))
acb_cosh_batch_prec_jit = jax.jit(acb_cosh_batch_prec, static_argnames=("prec_bits",))
acb_tanh_batch_prec_jit = jax.jit(acb_tanh_batch_prec, static_argnames=("prec_bits",))
acb_asin_batch_prec_jit = jax.jit(acb_asin_batch_prec, static_argnames=("prec_bits",))
acb_acos_batch_prec_jit = jax.jit(acb_acos_batch_prec, static_argnames=("prec_bits",))
acb_atan_batch_prec_jit = jax.jit(acb_atan_batch_prec, static_argnames=("prec_bits",))
acb_asinh_batch_prec_jit = jax.jit(acb_asinh_batch_prec, static_argnames=("prec_bits",))
acb_acosh_batch_prec_jit = jax.jit(acb_acosh_batch_prec, static_argnames=("prec_bits",))
acb_atanh_batch_prec_jit = jax.jit(acb_atanh_batch_prec, static_argnames=("prec_bits",))
acb_abs_batch_prec_jit = jax.jit(acb_abs_batch_prec, static_argnames=("prec_bits",))
acb_add_batch_prec_jit = jax.jit(acb_add_batch_prec, static_argnames=("prec_bits",))
acb_sub_batch_prec_jit = jax.jit(acb_sub_batch_prec, static_argnames=("prec_bits",))
acb_mul_batch_prec_jit = jax.jit(acb_mul_batch_prec, static_argnames=("prec_bits",))
acb_div_batch_prec_jit = jax.jit(acb_div_batch_prec, static_argnames=("prec_bits",))
acb_inv_batch_prec_jit = jax.jit(acb_inv_batch_prec, static_argnames=("prec_bits",))
acb_fma_batch_prec_jit = jax.jit(acb_fma_batch_prec, static_argnames=("prec_bits",))
acb_log1p_batch_prec_jit = jax.jit(acb_log1p_batch_prec, static_argnames=("prec_bits",))
acb_expm1_batch_prec_jit = jax.jit(acb_expm1_batch_prec, static_argnames=("prec_bits",))
acb_sin_cos_batch_prec_jit = jax.jit(acb_sin_cos_batch_prec, static_argnames=("prec_bits",))


__all__ = [
    "as_acb_box",
    "acb_box",
    "acb_real",
    "acb_imag",
    "acb_midpoint",
    "acb_zero",
    "acb_one",
    "acb_onei",
    "acb_is_exact",
    "acb_is_zero",
    "acb_is_one",
    "acb_is_real",
    "acb_is_int",
    "acb_set",
    "acb_neg",
    "acb_conj",
    "acb_add",
    "acb_sub",
    "acb_mul",
    "acb_mul_naive",
    "acb_mul_onei",
    "acb_div_onei",
    "acb_inv",
    "acb_div",
    "acb_abs",
    "acb_arg",
    "acb_sgn",
    "acb_csgn",
    "acb_real_abs",
    "acb_real_sgn",
    "acb_real_heaviside",
    "acb_real_floor",
    "acb_real_ceil",
    "acb_real_max",
    "acb_real_min",
    "acb_real_sqrtpos",
    "acb_sqrt_analytic",
    "acb_rsqrt_analytic",
    "acb_log_analytic",
    "acb_pow_analytic",
    "acb_dot_simple",
    "acb_dot_precise",
    "acb_dot",
    "acb_approx_dot",
    "acb_dot_ui",
    "acb_dot_si",
    "acb_dot_uiui",
    "acb_dot_siui",
    "acb_dot_fmpz",
    "acb_add_error_arb",
    "acb_add_error_arf",
    "acb_add_error_mag",
    "acb_union",
    "acb_trim",
    "acb_exp",
    "acb_log1p",
    "acb_log",
    "acb_sqrt",
    "acb_rsqrt",
    "acb_sin",
    "acb_cos",
    "acb_sin_cos",
    "acb_tan",
    "acb_cot",
    "acb_sinh",
    "acb_cosh",
    "acb_tanh",
    "acb_sech",
    "acb_csch",
    "acb_sin_pi",
    "acb_cos_pi",
    "acb_sin_cos_pi",
    "acb_tan_pi",
    "acb_cot_pi",
    "acb_csc_pi",
    "acb_sinc",
    "acb_sinc_pi",
    "acb_exp_pi_i",
    "acb_expm1",
    "acb_exp_invexp",
    "acb_addmul",
    "acb_submul",
    "acb_pow",
    "acb_pow_arb",
    "acb_pow_ui",
    "acb_pow_si",
    "acb_pow_fmpz",
    "acb_sqr",
    "acb_root_ui",
    "acb_gamma",
    "acb_rgamma",
    "acb_lgamma",
    "acb_log_sin_pi",
    "acb_digamma",
    "acb_zeta",
    "acb_hurwitz_zeta",
    "acb_polygamma",
    "acb_bernoulli_poly_ui",
    "acb_polylog",
    "acb_polylog_si",
    "acb_agm",
    "acb_agm1",
    "acb_agm1_cpx",
    "acb_log1p_prec",
    "acb_rsqrt_prec",
    "acb_sin_cos_prec",
    "acb_cot_prec",
    "acb_sech_prec",
    "acb_csch_prec",
    "acb_sin_pi_prec",
    "acb_cos_pi_prec",
    "acb_sin_cos_pi_prec",
    "acb_tan_pi_prec",
    "acb_cot_pi_prec",
    "acb_csc_pi_prec",
    "acb_sinc_prec",
    "acb_sinc_pi_prec",
    "acb_exp_pi_i_prec",
    "acb_expm1_prec",
    "acb_exp_invexp_prec",
    "acb_addmul_prec",
    "acb_submul_prec",
    "acb_pow_prec",
    "acb_pow_arb_prec",
    "acb_pow_ui_prec",
    "acb_pow_si_prec",
    "acb_pow_fmpz_prec",
    "acb_sqr_prec",
    "acb_root_ui_prec",
    "acb_gamma_prec",
    "acb_rgamma_prec",
    "acb_lgamma_prec",
    "acb_log_sin_pi_prec",
    "acb_digamma_prec",
    "acb_zeta_prec",
    "acb_hurwitz_zeta_prec",
    "acb_polygamma_prec",
    "acb_bernoulli_poly_ui_prec",
    "acb_polylog_prec",
    "acb_polylog_si_prec",
    "acb_agm_prec",
    "acb_agm1_prec",
    "acb_agm1_cpx_prec",
    "acb_exp_prec",
    "acb_log_prec",
    "acb_sqrt_prec",
    "acb_sin_prec",
    "acb_cos_prec",
    "acb_tan_prec",
    "acb_sinh_prec",
    "acb_cosh_prec",
    "acb_tanh_prec",
    "acb_abs_prec",
    "acb_add_prec",
    "acb_sub_prec",
    "acb_mul_prec",
    "acb_div_prec",
    "acb_inv_prec",
    "acb_fma_prec",
    "acb_exp_batch",
    "acb_log_batch",
    "acb_sqrt_batch",
    "acb_sin_batch",
    "acb_cos_batch",
    "acb_tan_batch",
    "acb_sinh_batch",
    "acb_cosh_batch",
    "acb_tanh_batch",
    "acb_abs_batch",
    "acb_add_batch",
    "acb_sub_batch",
    "acb_mul_batch",
    "acb_div_batch",
    "acb_inv_batch",
    "acb_fma_batch",
    "acb_log1p_batch",
    "acb_expm1_batch",
    "acb_sin_cos_batch",
    "acb_exp_batch_prec",
    "acb_log_batch_prec",
    "acb_sqrt_batch_prec",
    "acb_sin_batch_prec",
    "acb_cos_batch_prec",
    "acb_tan_batch_prec",
    "acb_sinh_batch_prec",
    "acb_cosh_batch_prec",
    "acb_tanh_batch_prec",
    "acb_abs_batch_prec",
    "acb_add_batch_prec",
    "acb_sub_batch_prec",
    "acb_mul_batch_prec",
    "acb_div_batch_prec",
    "acb_inv_batch_prec",
    "acb_fma_batch_prec",
    "acb_log1p_batch_prec",
    "acb_expm1_batch_prec",
    "acb_sin_cos_batch_prec",
    "acb_exp_batch_jit",
    "acb_log_batch_jit",
    "acb_sqrt_batch_jit",
    "acb_sin_batch_jit",
    "acb_cos_batch_jit",
    "acb_tan_batch_jit",
    "acb_sinh_batch_jit",
    "acb_cosh_batch_jit",
    "acb_tanh_batch_jit",
    "acb_abs_batch_jit",
    "acb_add_batch_jit",
    "acb_sub_batch_jit",
    "acb_mul_batch_jit",
    "acb_div_batch_jit",
    "acb_inv_batch_jit",
    "acb_fma_batch_jit",
    "acb_log1p_batch_jit",
    "acb_expm1_batch_jit",
    "acb_sin_cos_batch_jit",
    "acb_exp_batch_prec_jit",
    "acb_log_batch_prec_jit",
    "acb_sqrt_batch_prec_jit",
    "acb_sin_batch_prec_jit",
    "acb_cos_batch_prec_jit",
    "acb_tan_batch_prec_jit",
    "acb_sinh_batch_prec_jit",
    "acb_cosh_batch_prec_jit",
    "acb_tanh_batch_prec_jit",
    "acb_abs_batch_prec_jit",
    "acb_add_batch_prec_jit",
    "acb_sub_batch_prec_jit",
    "acb_mul_batch_prec_jit",
    "acb_div_batch_prec_jit",
    "acb_inv_batch_prec_jit",
    "acb_fma_batch_prec_jit",
    "acb_log1p_batch_prec_jit",
    "acb_expm1_batch_prec_jit",
    "acb_sin_cos_batch_prec_jit",
]

__all__.extend(
    [
        "acb_barnes_g",
        "acb_log_barnes_g",
        "acb_barnes_g_prec",
        "acb_log_barnes_g_prec",
        "acb_asin",
        "acb_acos",
        "acb_atan",
        "acb_asinh",
        "acb_acosh",
        "acb_atanh",
        "acb_asin_prec",
        "acb_acos_prec",
        "acb_atan_prec",
        "acb_asinh_prec",
        "acb_acosh_prec",
        "acb_atanh_prec",
        "acb_asin_batch",
        "acb_acos_batch",
        "acb_atan_batch",
        "acb_asinh_batch",
        "acb_acosh_batch",
        "acb_atanh_batch",
        "acb_asin_batch_prec",
        "acb_acos_batch_prec",
        "acb_atan_batch_prec",
        "acb_asinh_batch_prec",
        "acb_acosh_batch_prec",
        "acb_atanh_batch_prec",
        "acb_asin_batch_jit",
        "acb_acos_batch_jit",
        "acb_atan_batch_jit",
        "acb_asinh_batch_jit",
        "acb_acosh_batch_jit",
        "acb_atanh_batch_jit",
        "acb_asin_batch_prec_jit",
        "acb_acos_batch_prec_jit",
        "acb_atan_batch_prec_jit",
        "acb_asinh_batch_prec_jit",
        "acb_acosh_batch_prec_jit",
        "acb_atanh_batch_prec_jit",
    ]
)
