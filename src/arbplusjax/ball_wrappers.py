from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy.special as jsp

from . import double_interval as di
from . import precision
from . import acb_core
from . import barnesg
from . import double_gamma

jax.config.update("jax_enable_x64", True)

_LANCZOS = jnp.asarray(
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
    dtype=jnp.float64,
)


def _full_interval() -> jax.Array:
    return di.interval(-jnp.inf, jnp.inf)


def _full_box() -> jax.Array:
    return jnp.array([-jnp.inf, jnp.inf, -jnp.inf, jnp.inf], dtype=jnp.float64)


def _ball_from_interval(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = di.as_interval(x)
    mid = di.midpoint(x)
    rad = 0.5 * (x[1] - x[0])
    return mid, jnp.maximum(rad, 0.0)


def _box_from_ball(mid: jax.Array, rad: jax.Array) -> jax.Array:
    rad = jnp.maximum(rad, 0.0)
    return jnp.array(
        [jnp.real(mid) - rad, jnp.real(mid) + rad, jnp.imag(mid) - rad, jnp.imag(mid) + rad],
        dtype=jnp.float64,
    )


def _ball_from_box(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    box = acb_core.as_acb_box(x)
    re = acb_core.acb_real(box)
    im = acb_core.acb_imag(box)
    mid = di.midpoint(re) + 1j * di.midpoint(im)
    rad = jnp.maximum(0.5 * (re[1] - re[0]), 0.5 * (im[1] - im[0]))
    return mid, rad


def _map_interval(fn, x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    if x.ndim == 1:
        return fn(x)
    return jax.vmap(fn)(x)


def _map_interval_pair(fn, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = di.as_interval(x)
    if x.ndim == 1:
        return fn(x)
    return jax.vmap(fn)(x)


def _map_interval_bivariate(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    y = di.as_interval(y)
    if x.ndim == 1 and y.ndim == 1:
        return fn(x, y)
    return jax.vmap(fn)(x, y)


def _map_box_bivariate(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    x = acb_core.as_acb_box(x)
    y = acb_core.as_acb_box(y)
    if x.ndim == 1 and y.ndim == 1:
        return fn(x, y)
    return jax.vmap(fn)(x, y)


def _map_box(fn, x: jax.Array) -> jax.Array:
    x = acb_core.as_acb_box(x)
    if x.ndim == 1:
        return fn(x)
    return jax.vmap(fn)(x)


def _real_deriv_bound(fn, x: jax.Array) -> jax.Array:
    return jnp.abs(jax.grad(fn)(x))


def _complex_deriv_bound(fn, z: jax.Array) -> jax.Array:
    def f_xy(xy):
        x, y = xy[0], xy[1]
        w = fn(x + 1j * y)
        return jnp.array([jnp.real(w), jnp.imag(w)])

    j = jax.jacfwd(f_xy)(jnp.array([jnp.real(z), jnp.imag(z)], dtype=jnp.float64))
    return jnp.sqrt(jnp.sum(j * j))


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = z - jnp.complex128(1.0 + 0.0j)
    x = jnp.complex128(_LANCZOS[0] + 0.0j)

    def body(i, acc):
        return acc + _LANCZOS[i] / (z1 + jnp.float64(i))

    x = lax.fori_loop(1, 9, body, x)
    t = z1 + jnp.float64(7.5)
    return jnp.float64(0.91893853320467274178) + (z1 + 0.5) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)

    def reflection(w):
        return jnp.log(jnp.pi) - jnp.log(jnp.sin(jnp.pi * w)) - _complex_loggamma_lanczos(1.0 - w)

    return lax.cond(jnp.real(z) < 0.5, reflection, _complex_loggamma_lanczos, z)


_LIP_SAMPLES = 9


def _rigorous_real(fn, x: jax.Array, eps: float) -> jax.Array:
    mid, rad = _ball_from_interval(x)
    val = fn(mid)
    ts = jnp.linspace(-1.0, 1.0, _LIP_SAMPLES)
    xs = mid + rad * ts
    L = jnp.max(jax.vmap(lambda t: _real_deriv_bound(fn, t))(xs))
    rad_out = L * rad + eps
    out = di.interval(di._below(val - rad_out), di._above(val + rad_out))
    return jnp.where(jnp.isfinite(val), out, _full_interval())


def _rigorous_complex(fn, x: jax.Array, eps: float) -> jax.Array:
    mid, rad = _ball_from_box(x)
    val = fn(mid)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, _LIP_SAMPLES, endpoint=False)
    zs = mid + rad * (jnp.cos(angles) + 1j * jnp.sin(angles))
    L = jnp.max(jax.vmap(lambda t: _complex_deriv_bound(fn, t))(zs))
    rad_out = L * rad + eps
    finite = jnp.isfinite(jnp.real(val)) & jnp.isfinite(jnp.imag(val))
    out = _box_from_ball(val, rad_out)
    return jnp.where(finite, out, _full_box())


def _adaptive_real(fn, x: jax.Array, eps: float, samples: int) -> jax.Array:
    mid, rad = _ball_from_interval(x)
    t = jnp.linspace(-1.0, 1.0, samples)
    xs = mid + rad * t
    vals = jax.vmap(fn)(xs)
    v0 = fn(mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = di.interval(di._below(v0 - rad_out), di._above(v0 + rad_out))
    return jnp.where(jnp.isfinite(v0), out, _full_interval())


def _adaptive_complex(fn, x: jax.Array, eps: float, samples: int) -> jax.Array:
    mid, rad = _ball_from_box(x)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, samples, endpoint=False)
    zs = mid + rad * (jnp.cos(angles) + 1j * jnp.sin(angles))
    vals = jax.vmap(fn)(zs)
    v0 = fn(mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    finite = jnp.isfinite(jnp.real(v0)) & jnp.isfinite(jnp.imag(v0))
    out = _box_from_ball(v0, rad_out)
    return jnp.where(finite, out, _full_box())


def _rigorous_real_bivariate(fn, x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    x_mid, x_rad = _ball_from_interval(x)
    y_mid, y_rad = _ball_from_interval(y)
    val = fn(x_mid, y_mid)

    def fx(v):
        return fn(v, y_mid)

    def fy(v):
        return fn(x_mid, v)

    dx = _real_deriv_bound(fx, x_mid)
    dy = _real_deriv_bound(fy, y_mid)
    rad_out = dx * x_rad + dy * y_rad + eps
    out = di.interval(di._below(val - rad_out), di._above(val + rad_out))
    return jnp.where(jnp.isfinite(val), out, _full_interval())


def _adaptive_real_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    x_mid, x_rad = _ball_from_interval(x)
    y_mid, y_rad = _ball_from_interval(y)
    xs = x_mid + x_rad * jnp.linspace(-1.0, 1.0, samples)
    ys = y_mid + y_rad * jnp.linspace(-1.0, 1.0, samples)
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(ys))(xs).reshape(-1)
    v0 = fn(x_mid, y_mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = di.interval(di._below(v0 - rad_out), di._above(v0 + rad_out))
    return jnp.where(jnp.isfinite(v0), out, _full_interval())


def _rigorous_complex_bivariate(fn, x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    x_mid, x_rad = _ball_from_box(x)
    y_mid, y_rad = _ball_from_box(y)
    val = fn(x_mid, y_mid)

    def f_xy(vec):
        xr, xi, yr, yi = vec
        w = fn(xr + 1j * xi, yr + 1j * yi)
        return jnp.array([jnp.real(w), jnp.imag(w)])

    vec0 = jnp.array([jnp.real(x_mid), jnp.imag(x_mid), jnp.real(y_mid), jnp.imag(y_mid)], dtype=jnp.float64)
    jac = jax.jacfwd(f_xy)(vec0)
    rad_vec = jnp.array([x_rad, x_rad, y_rad, y_rad], dtype=jnp.float64)
    bound_re = jnp.sum(jnp.abs(jac[0]) * rad_vec)
    bound_im = jnp.sum(jnp.abs(jac[1]) * rad_vec)
    re_iv = di.interval(di._below(jnp.real(val) - bound_re), di._above(jnp.real(val) + bound_re))
    im_iv = di.interval(di._below(jnp.imag(val) - bound_im), di._above(jnp.imag(val) + bound_im))
    finite = jnp.isfinite(jnp.real(val)) & jnp.isfinite(jnp.imag(val))
    out = acb_core.acb_box(re_iv, im_iv)
    return jnp.where(finite, out, _full_box())


def _adaptive_complex_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    x_mid, x_rad = _ball_from_box(x)
    y_mid, y_rad = _ball_from_box(y)
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, samples, endpoint=False)
    xs = x_mid + x_rad * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ys = y_mid + y_rad * (jnp.cos(angles) + 1j * jnp.sin(angles))
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(ys))(xs).reshape(-1)
    v0 = fn(x_mid, y_mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = _box_from_ball(v0, rad_out)
    finite = jnp.isfinite(jnp.real(v0)) & jnp.isfinite(jnp.imag(v0))
    return jnp.where(finite, out, _full_box())


def _real_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    nu = jnp.asarray(nu, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    half = 0.5 * z
    term0 = jnp.power(half, nu) / jnp.exp(jsp.gammaln(nu + 1.0))
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        den = k1 * (k1 + nu)
        num = 0.25 * sign * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, 59, body, (term0, sum0))
    return s


def _complex_bessel_series(nu: jax.Array, z: jax.Array, sign: float) -> jax.Array:
    nu = jnp.asarray(nu, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    half = 0.5 * z
    pow_half = jnp.exp(nu * jnp.log(half))
    gamma = jnp.exp(_complex_loggamma(nu + 1.0))
    term0 = pow_half / gamma
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        den = k1 * (nu + k1)
        num = (0.25 * sign) * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, 59, body, (term0, sum0))
    return s


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_exp(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.exp, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_log(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.log, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_sin(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.sin, x, eps)


def _contains_nonpositive_integer_interval(x: jax.Array) -> jax.Array:
    x = di.as_interval(x)
    return (x[0] <= 0.0) & (jnp.floor(x[1]) <= 0.0) & (jnp.floor(x[1]) >= x[0])


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_barnesg(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    cross = _contains_nonpositive_integer_interval(x)
    return lax.cond(cross, lambda _: _full_interval(), lambda _: _rigorous_real(lambda t: barnesg.barnesg_real(t), x, eps), operand=None)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_barnesg_adaptive(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = di.as_interval(x)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    cross = _contains_nonpositive_integer_interval(x)
    return lax.cond(cross, lambda _: _full_interval(), lambda _: _adaptive_real(lambda t: barnesg.barnesg_real(t), x, eps, samples=_LIP_SAMPLES), operand=None)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_barnesg(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = acb_core.as_acb_box(x)
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    cross_pole = (im[0] <= 0.0) & (im[1] >= 0.0) & (re[0] <= 0.0) & (jnp.floor(re[1]) <= 0.0) & (jnp.floor(re[1]) >= re[0])
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return lax.cond(cross_pole, lambda _: _full_box(), lambda _: _rigorous_complex(lambda z: barnesg.barnesg_complex(z), x, eps), operand=None)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_barnesg_adaptive(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    x = acb_core.as_acb_box(x)
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    cross_pole = (im[0] <= 0.0) & (im[1] >= 0.0) & (re[0] <= 0.0) & (jnp.floor(re[1]) <= 0.0) & (jnp.floor(re[1]) >= re[0])
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return lax.cond(cross_pole, lambda _: _full_box(), lambda _: _adaptive_complex(lambda z: barnesg.barnesg_complex(z), x, eps, samples=_LIP_SAMPLES), operand=None)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = di.as_interval(z)
    tau = di.as_interval(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = di.midpoint(tau)
    return _rigorous_real(lambda zz: jnp.real(double_gamma.log_barnesdoublegamma(zz, tau_mid, prec_bits)), z, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_log_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = di.as_interval(z)
    tau = di.as_interval(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = di.midpoint(tau)
    return _adaptive_real(lambda zz: jnp.real(double_gamma.log_barnesdoublegamma(zz, tau_mid, prec_bits)), z, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = di.as_interval(z)
    tau = di.as_interval(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = di.midpoint(tau)
    return _rigorous_real(lambda zz: jnp.real(double_gamma.barnesdoublegamma(zz, tau_mid, prec_bits)), z, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = di.as_interval(z)
    tau = di.as_interval(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = di.midpoint(tau)
    return _adaptive_real(lambda zz: jnp.real(double_gamma.barnesdoublegamma(zz, tau_mid, prec_bits)), z, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_loggamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _rigorous_real(lambda ww: jnp.real(double_gamma.loggamma2(ww, beta_mid, prec_bits)), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_loggamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _adaptive_real(lambda ww: jnp.real(double_gamma.loggamma2(ww, beta_mid, prec_bits)), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_gamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _rigorous_real(lambda ww: jnp.real(double_gamma.gamma2(ww, beta_mid, prec_bits)), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_gamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _adaptive_real(lambda ww: jnp.real(double_gamma.gamma2(ww, beta_mid, prec_bits)), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_logdoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _rigorous_real(lambda ww: jnp.real(double_gamma.logdoublegamma(ww, beta_mid, prec_bits)), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_logdoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _adaptive_real(lambda ww: jnp.real(double_gamma.logdoublegamma(ww, beta_mid, prec_bits)), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_doublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _rigorous_real(lambda ww: jnp.real(double_gamma.doublegamma(ww, beta_mid, prec_bits)), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_doublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = di.as_interval(w)
    beta = di.as_interval(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = di.midpoint(beta)
    return _adaptive_real(lambda ww: jnp.real(double_gamma.doublegamma(ww, beta_mid, prec_bits)), w, eps, samples=_LIP_SAMPLES)

@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    tau = acb_core.as_acb_box(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = acb_core.acb_midpoint(tau)
    return _rigorous_complex(lambda zz: double_gamma.log_barnesdoublegamma(zz, tau_mid, prec_bits), z, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_log_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    tau = acb_core.as_acb_box(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = acb_core.acb_midpoint(tau)
    return _adaptive_complex(lambda zz: double_gamma.log_barnesdoublegamma(zz, tau_mid, prec_bits), z, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    tau = acb_core.as_acb_box(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = acb_core.acb_midpoint(tau)
    return _rigorous_complex(lambda zz: double_gamma.barnesdoublegamma(zz, tau_mid, prec_bits), z, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    tau = acb_core.as_acb_box(tau)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    tau_mid = acb_core.acb_midpoint(tau)
    return _adaptive_complex(lambda zz: double_gamma.barnesdoublegamma(zz, tau_mid, prec_bits), z, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_loggamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _rigorous_complex(lambda ww: double_gamma.loggamma2(ww, beta_mid, prec_bits), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_loggamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _adaptive_complex(lambda ww: double_gamma.loggamma2(ww, beta_mid, prec_bits), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_gamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _rigorous_complex(lambda ww: double_gamma.gamma2(ww, beta_mid, prec_bits), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_gamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _adaptive_complex(lambda ww: double_gamma.gamma2(ww, beta_mid, prec_bits), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_logdoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _rigorous_complex(lambda ww: double_gamma.logdoublegamma(ww, beta_mid, prec_bits), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_logdoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _adaptive_complex(lambda ww: double_gamma.logdoublegamma(ww, beta_mid, prec_bits), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_doublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _rigorous_complex(lambda ww: double_gamma.doublegamma(ww, beta_mid, prec_bits), w, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_doublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    w = acb_core.as_acb_box(w)
    beta = acb_core.as_acb_box(beta)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    beta_mid = acb_core.acb_midpoint(beta)
    return _adaptive_complex(lambda ww: double_gamma.doublegamma(ww, beta_mid, prec_bits), w, eps, samples=_LIP_SAMPLES)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    b = acb_core.as_acb_box(b)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    b_mid = acb_core.acb_midpoint(b)
    return _rigorous_complex(lambda zz: double_gamma.double_sine(zz, b_mid, prec_bits), z, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_double_sine_adaptive(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    z = acb_core.as_acb_box(z)
    b = acb_core.as_acb_box(b)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    b_mid = acb_core.acb_midpoint(b)
    return _adaptive_complex(lambda zz: double_gamma.double_sine(zz, b_mid, prec_bits), z, eps, samples=_LIP_SAMPLES)


def arb_ball_gamma(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jsp.gamma, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_exp(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_complex(jnp.exp, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_log(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_complex(jnp.log, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_sin(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_complex(jnp.sin, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_gamma(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_complex(lambda z: jnp.exp(acb_core._complex_loggamma(z)), x, eps)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_exp_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.exp, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_log_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.log, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_sin_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.sin, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_gamma_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jsp.gamma, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_exp_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_complex(jnp.exp, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_log_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_complex(jnp.log, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_sin_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_complex(jnp.sin, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_gamma_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_complex(lambda z: jnp.exp(acb_core._complex_loggamma(z)), x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erf(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.erf, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfc(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.erfc, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.erfi, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfinv(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.erfinv, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfcinv(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: jsp.erfinv(1.0 - v), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erf(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(jsp.erf, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erfc(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(lambda z: 1.0 - jsp.erf(z), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erfi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(lambda z: -1j * jsp.erf(1j * z), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_ei(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.expi, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_si(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: jsp.sici(v)[0], t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_ci(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: jsp.sici(v)[1], t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_shi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.shi, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_chi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(jsp.chi, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_li(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        out = _rigorous_real(lambda v: jsp.expi(jnp.log(v)), t, eps)
        full = _full_interval()
        return jnp.where(t[0] <= 0.0, full, out)

    return _map_interval(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_dilog(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: jsp.spence(1.0 - v), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_fresnel(x: jax.Array, prec_bits: int = 53, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        s = _rigorous_real(lambda v: jsp.fresnel(v)[0], t, eps)
        c = _rigorous_real(lambda v: jsp.fresnel(v)[1], t, eps)
        if not normalized:
            s = di.fast_mul(s, di.interval(jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0))))
            c = di.fast_mul(c, di.interval(jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0))))
        return s, c

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_airy(x: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        ai = _rigorous_real(lambda v: jsp.airy(v)[0], t, eps)
        aip = _rigorous_real(lambda v: jsp.airy(v)[1], t, eps)
        bi = _rigorous_real(lambda v: jsp.airy(v)[2], t, eps)
        bip = _rigorous_real(lambda v: jsp.airy(v)[3], t, eps)
        return ai, aip, bi, bip

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erf_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.erf, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfc_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.erfc, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.erfi, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfinv_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.erfinv, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfcinv_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jsp.erfinv(1.0 - v), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erf_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(jsp.erf, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erfc_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(lambda z: 1.0 - jsp.erf(z), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erfi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(lambda z: -1j * jsp.erf(1j * z), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_ei_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.expi, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_si_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jsp.sici(v)[0], t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_ci_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jsp.sici(v)[1], t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_shi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.shi, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_chi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(jsp.chi, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_li_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        out = _adaptive_real(lambda v: jsp.expi(jnp.log(v)), t, eps, samples)
        full = _full_interval()
        return jnp.where(t[0] <= 0.0, full, out)

    return _map_interval(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_dilog_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jsp.spence(1.0 - v), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_fresnel_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        s = _adaptive_real(lambda v: jsp.fresnel(v)[0], t, eps, samples)
        c = _adaptive_real(lambda v: jsp.fresnel(v)[1], t, eps, samples)
        if not normalized:
            s = di.fast_mul(s, di.interval(jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0))))
            c = di.fast_mul(c, di.interval(jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(jnp.pi) / jnp.sqrt(2.0))))
        return s, c

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_airy_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        ai = _adaptive_real(lambda v: jsp.airy(v)[0], t, eps, samples)
        aip = _adaptive_real(lambda v: jsp.airy(v)[1], t, eps, samples)
        bi = _adaptive_real(lambda v: jsp.airy(v)[2], t, eps, samples)
        bip = _adaptive_real(lambda v: jsp.airy(v)[3], t, eps, samples)
        return ai, aip, bi, bip

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_j(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(lambda u, v: _real_bessel_series(u, v, -1.0), a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_i(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(lambda u, v: _real_bessel_series(u, v, 1.0), a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_y(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        jnu = _real_bessel_series(u, v, -1.0)
        jneg = _real_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(jnp.pi * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_k(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _real_bessel_series(u, v, 1.0)
        ineg = _real_bessel_series(-u, v, 1.0)
        val = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_jy(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array]:
    return arb_ball_bessel_j(nu, z, prec_bits), arb_ball_bessel_y(nu, z, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_i_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _real_bessel_series(u, v, 1.0)

    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_k_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _real_bessel_series(u, v, 1.0)
        ineg = _real_bessel_series(-u, v, 1.0)
        k = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_interval_bivariate(lambda a, b: _rigorous_real_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_j_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(lambda u, v: _real_bessel_series(u, v, -1.0), a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_i_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(lambda u, v: _real_bessel_series(u, v, 1.0), a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_y_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        jnu = _real_bessel_series(u, v, -1.0)
        jneg = _real_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(jnp.pi * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_k_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _real_bessel_series(u, v, 1.0)
        ineg = _real_bessel_series(-u, v, 1.0)
        val = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_jy_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array]:
    return arb_ball_bessel_j_adaptive(nu, z, prec_bits, samples), arb_ball_bessel_y_adaptive(nu, z, prec_bits, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_i_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _real_bessel_series(u, v, 1.0)

    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_k_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _real_bessel_series(u, v, 1.0)
        ineg = _real_bessel_series(-u, v, 1.0)
        k = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_interval_bivariate(lambda a, b: _adaptive_real_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_j(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(lambda u, v: _complex_bessel_series(u, v, -1.0), a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_i(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(lambda u, v: _complex_bessel_series(u, v, 1.0), a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_y(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        jnu = _complex_bessel_series(u, v, -1.0)
        jneg = _complex_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(jnp.pi * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_k(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        val = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_jy(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array]:
    return acb_ball_bessel_j(nu, z, prec_bits), acb_ball_bessel_y(nu, z, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_i_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _complex_bessel_series(u, v, 1.0)

    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_k_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        k = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_box_bivariate(lambda a, b: _rigorous_complex_bivariate(fn, a, b, eps), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_j_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(lambda u, v: _complex_bessel_series(u, v, -1.0), a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_i_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(lambda u, v: _complex_bessel_series(u, v, 1.0), a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_y_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        jnu = _complex_bessel_series(u, v, -1.0)
        jneg = _complex_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(jnp.pi * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_k_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        val = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_jy_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array]:
    return acb_ball_bessel_j_adaptive(nu, z, prec_bits, samples), acb_ball_bessel_y_adaptive(nu, z, prec_bits, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_i_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _complex_bessel_series(u, v, 1.0)

    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(fn, a, b, eps, samples), nu, z)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_k_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(jnp.pi * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        k = 0.5 * jnp.pi * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(fn, a, b, eps, samples), nu, z)


def arb_ball_exp_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_exp(x, prec_bits=prec_bits)


def arb_ball_log_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_log(x, prec_bits=prec_bits)


def arb_ball_sin_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_sin(x, prec_bits=prec_bits)


def arb_ball_gamma_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_gamma(x, prec_bits=prec_bits)


def acb_ball_exp_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return acb_ball_exp(x, prec_bits=prec_bits)


def acb_ball_log_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return acb_ball_log(x, prec_bits=prec_bits)


def acb_ball_sin_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return acb_ball_sin(x, prec_bits=prec_bits)


def acb_ball_gamma_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return acb_ball_gamma(x, prec_bits=prec_bits)


__all__ = [
    "arb_ball_exp",
    "arb_ball_log",
    "arb_ball_sin",
    "arb_ball_gamma",
    "acb_ball_exp",
    "acb_ball_log",
    "acb_ball_sin",
    "acb_ball_gamma",
    "arb_ball_exp_adaptive",
    "arb_ball_log_adaptive",
    "arb_ball_sin_adaptive",
    "arb_ball_gamma_adaptive",
    "acb_ball_exp_adaptive",
    "acb_ball_log_adaptive",
    "acb_ball_sin_adaptive",
    "acb_ball_gamma_adaptive",
    "arb_ball_exp_mp",
    "arb_ball_log_mp",
    "arb_ball_sin_mp",
    "arb_ball_gamma_mp",
    "acb_ball_exp_mp",
    "acb_ball_log_mp",
    "acb_ball_sin_mp",
    "acb_ball_gamma_mp",
]
