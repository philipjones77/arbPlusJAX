from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from . import double_interval as di
from . import arb_core
from . import acb_dirichlet
from . import hypgeom

jax.config.update("jax_enable_x64", True)


def as_acb_box(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got shape {arr.shape}")
    return arr


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
    z = jnp.float64(0.0)
    return acb_box(di.interval(z, z), di.interval(z, z))


def acb_one() -> jax.Array:
    o = jnp.float64(1.0)
    z = jnp.float64(0.0)
    return acb_box(di.interval(o, o), di.interval(z, z))


def acb_onei() -> jax.Array:
    o = jnp.float64(1.0)
    z = jnp.float64(0.0)
    return acb_box(di.interval(z, z), di.interval(o, o))


def _full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.float64)
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
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = z - jnp.complex128(1.0 + 0.0j)
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
        dtype=jnp.complex128,
    )
    x = coeffs[0]
    for k in range(1, coeffs.shape[0]):
        x = x + coeffs[k] / (z1 + jnp.float64(k))
    t = z1 + jnp.float64(7.5)
    return jnp.float64(0.91893853320467274178) + (z1 + 0.5) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)

    def reflection(w):
        return jnp.log(jnp.pi) - jnp.log(jnp.sin(jnp.pi * w)) - _complex_loggamma_lanczos(1.0 - w)

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
    yb = acb_box(di.interval(jnp.asarray(y, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), di.interval(0.0, 0.0))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_dot_si(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    yb = acb_box(di.interval(jnp.asarray(y, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), di.interval(0.0, 0.0))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_dot_uiui(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_ui(initial, subtract, x, y)


def acb_dot_siui(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    return acb_dot_ui(initial, subtract, x, y)


def acb_dot_fmpz(initial: jax.Array, subtract: int, x: jax.Array, y: jax.Array) -> jax.Array:
    yb = acb_box(di.interval(jnp.asarray(y, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), di.interval(0.0, 0.0))
    return acb_dot_simple(initial, subtract, x, yb)


def acb_add_error_arb(x: jax.Array, err: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    er = di.as_interval(err)
    return acb_box(di.fast_add(acb_real(xb), er), di.fast_add(acb_imag(xb), er))


def acb_add_error_arf(x: jax.Array, err: jax.Array) -> jax.Array:
    e = jnp.asarray(err, dtype=jnp.float64)
    er = di.interval(e, e)
    return acb_add_error_arb(x, er)


def acb_add_error_mag(x: jax.Array, err: jax.Array) -> jax.Array:
    e = jnp.asarray(err, dtype=jnp.float64)
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
def acb_sech(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.cosh(z))


@jax.jit
def acb_csch(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.sinh(z))


@jax.jit
def acb_sin_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.sin(jnp.pi * z))


@jax.jit
def acb_cos_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.cos(jnp.pi * z))


@jax.jit
def acb_sin_cos_pi(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _acb_unary_pair_from_midpoint(x, lambda z: (jnp.sin(jnp.pi * z), jnp.cos(jnp.pi * z)))


@jax.jit
def acb_tan_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.tan(jnp.pi * z))


@jax.jit
def acb_cot_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.tan(jnp.pi * z))


@jax.jit
def acb_csc_pi(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: 1.0 / jnp.sin(jnp.pi * z))


@jax.jit
def acb_sinc(x: jax.Array) -> jax.Array:
    def fn(z):
        return jnp.where(jnp.abs(z) < 1e-12, 1.0 + 0.0j, jnp.sin(z) / z)

    return _acb_unary_from_midpoint(x, fn)


@jax.jit
def acb_sinc_pi(x: jax.Array) -> jax.Array:
    def fn(z):
        t = jnp.pi * z
        return jnp.where(jnp.abs(z) < 1e-12, 1.0 + 0.0j, jnp.sin(t) / t)

    return _acb_unary_from_midpoint(x, fn)


@jax.jit
def acb_exp_pi_i(x: jax.Array) -> jax.Array:
    return _acb_unary_from_midpoint(x, lambda z: jnp.exp(1j * jnp.pi * z))


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
    v = jnp.exp(w * jnp.log(z))
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(xb))


@jax.jit
def acb_pow_arb(x: jax.Array, y: jax.Array) -> jax.Array:
    yb = di.as_interval(y)
    m = di.midpoint(yb)
    return acb_pow(x, acb_box(di.interval(m, m), di.interval(jnp.float64(0.0), jnp.float64(0.0))))


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
    return _acb_unary_from_midpoint(x, lambda z: jnp.power(z, 1.0 / jnp.float64(k)))


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
    return _acb_unary_from_midpoint(x, lambda z: jnp.log(jnp.sin(jnp.pi * z)))


@jax.jit
def acb_digamma(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    z = acb_midpoint(box)
    h = jnp.complex128(1e-6 + 0.0j)
    v = (_complex_loggamma(z + h) - _complex_loggamma(z - h)) / (2.0 * h)
    finite = jnp.isfinite(jnp.real(v)) & jnp.isfinite(jnp.imag(v))
    out = _acb_from_complex(v)
    return jnp.where(finite[..., None], out, _full_box_like(box))


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

    if n == 0:
        v = jnp.complex128(1.0 + 0.0j)
    elif n == 1:
        v = z - 0.5
    elif n == 2:
        v = z * z - z + jnp.float64(1.0 / 6.0)
    elif n == 3:
        v = z * z * z - 1.5 * z * z + 0.5 * z
    elif n == 4:
        v = z**4 - 2.0 * z**3 + z * z - jnp.float64(1.0 / 30.0)
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
    s_box = acb_box(
        di.interval(jnp.float64(s), jnp.float64(s)),
        di.interval(jnp.float64(0.0), jnp.float64(0.0)),
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


acb_exp_batch_jit = jax.jit(acb_exp_batch)
acb_log_batch_jit = jax.jit(acb_log_batch)
acb_sqrt_batch_jit = jax.jit(acb_sqrt_batch)
acb_sin_batch_jit = jax.jit(acb_sin_batch)
acb_cos_batch_jit = jax.jit(acb_cos_batch)
acb_tan_batch_jit = jax.jit(acb_tan_batch)
acb_sinh_batch_jit = jax.jit(acb_sinh_batch)
acb_cosh_batch_jit = jax.jit(acb_cosh_batch)
acb_tanh_batch_jit = jax.jit(acb_tanh_batch)

acb_exp_batch_prec_jit = jax.jit(acb_exp_batch_prec, static_argnames=("prec_bits",))
acb_log_batch_prec_jit = jax.jit(acb_log_batch_prec, static_argnames=("prec_bits",))
acb_sqrt_batch_prec_jit = jax.jit(acb_sqrt_batch_prec, static_argnames=("prec_bits",))
acb_sin_batch_prec_jit = jax.jit(acb_sin_batch_prec, static_argnames=("prec_bits",))
acb_cos_batch_prec_jit = jax.jit(acb_cos_batch_prec, static_argnames=("prec_bits",))
acb_tan_batch_prec_jit = jax.jit(acb_tan_batch_prec, static_argnames=("prec_bits",))
acb_sinh_batch_prec_jit = jax.jit(acb_sinh_batch_prec, static_argnames=("prec_bits",))
acb_cosh_batch_prec_jit = jax.jit(acb_cosh_batch_prec, static_argnames=("prec_bits",))
acb_tanh_batch_prec_jit = jax.jit(acb_tanh_batch_prec, static_argnames=("prec_bits",))


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
    "acb_exp_prec",
    "acb_log_prec",
    "acb_sqrt_prec",
    "acb_sin_prec",
    "acb_cos_prec",
    "acb_tan_prec",
    "acb_sinh_prec",
    "acb_cosh_prec",
    "acb_tanh_prec",
    "acb_exp_batch",
    "acb_log_batch",
    "acb_sqrt_batch",
    "acb_sin_batch",
    "acb_cos_batch",
    "acb_tan_batch",
    "acb_sinh_batch",
    "acb_cosh_batch",
    "acb_tanh_batch",
    "acb_exp_batch_prec",
    "acb_log_batch_prec",
    "acb_sqrt_batch_prec",
    "acb_sin_batch_prec",
    "acb_cos_batch_prec",
    "acb_tan_batch_prec",
    "acb_sinh_batch_prec",
    "acb_cosh_batch_prec",
    "acb_tanh_batch_prec",
    "acb_exp_batch_jit",
    "acb_log_batch_jit",
    "acb_sqrt_batch_jit",
    "acb_sin_batch_jit",
    "acb_cos_batch_jit",
    "acb_tan_batch_jit",
    "acb_sinh_batch_jit",
    "acb_cosh_batch_jit",
    "acb_tanh_batch_jit",
    "acb_exp_batch_prec_jit",
    "acb_log_batch_prec_jit",
    "acb_sqrt_batch_prec_jit",
    "acb_sin_batch_prec_jit",
    "acb_cos_batch_prec_jit",
    "acb_tan_batch_prec_jit",
    "acb_sinh_batch_prec_jit",
    "acb_cosh_batch_prec_jit",
    "acb_tanh_batch_prec_jit",
]
