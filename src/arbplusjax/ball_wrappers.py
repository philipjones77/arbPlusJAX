from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import double_interval as di
from . import precision
from . import acb_core
from . import arb_core
from . import barnesg
from . import coeffs
from . import double_gamma
from . import elementary as el

jax.config.update("jax_enable_x64", True)


def _gamma_real(x: jax.Array) -> jax.Array:
    return jnp.exp(lax.lgamma(x))


def _expi_real(x: jax.Array) -> jax.Array:
    from . import hypgeom
    return jnp.real(hypgeom._complex_ei_series(jnp.asarray(x, dtype=jnp.complex128)))


def _dilog_real(x: jax.Array) -> jax.Array:
    from . import hypgeom
    return jnp.real(hypgeom._complex_dilog_series(jnp.asarray(x, dtype=jnp.complex128)))


def _erfinv_real(x: jax.Array) -> jax.Array:
    from . import hypgeom
    return hypgeom._real_erfinv_scalar(x)


def _erfi_real(x: jax.Array) -> jax.Array:
    from . import hypgeom
    return jnp.real(hypgeom._complex_erfi_series(jnp.asarray(x, dtype=jnp.complex128)))


def _erf_complex(z: jax.Array) -> jax.Array:
    from . import hypgeom
    return hypgeom._complex_erf_series(jnp.asarray(z, dtype=jnp.complex128))


def _si_ci_real(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    from . import hypgeom
    use_series = jnp.abs(x) <= 4.0
    s_series, c_series = hypgeom._si_ci_from_series(x)
    s_asymp, c_asymp = hypgeom._si_ci_asymp(x)
    s = jnp.where(use_series, s_series, s_asymp)
    c = jnp.where(use_series, c_series, c_asymp)
    return s, c


def _fresnel_real(x: jax.Array, normalized: bool) -> tuple[jax.Array, jax.Array]:
    from . import hypgeom
    s, c = hypgeom._complex_fresnel(jnp.asarray(x, dtype=jnp.complex128), True)
    if not normalized:
        s = hypgeom._SQRT_PI_OVER_2 * s
        c = hypgeom._SQRT_PI_OVER_2 * c
    return s, c


def _airy_real(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    from . import hypgeom
    ai, aip = hypgeom._airy_series(x, -1.0)
    bi, bip = hypgeom._airy_series(x, 1.0)
    return ai, aip, bi, bip

_LANCZOS = coeffs.LANCZOS


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


def _complex_partial_bounds_bivariate(fn, z: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
    def f_xyzw(xyzw):
        zc = xyzw[0] + 1j * xyzw[1]
        wc = xyzw[2] + 1j * xyzw[3]
        out = fn(zc, wc)
        return jnp.array([jnp.real(out), jnp.imag(out)], dtype=jnp.float64)

    xyzw = jnp.array([jnp.real(z), jnp.imag(z), jnp.real(w), jnp.imag(w)], dtype=jnp.float64)
    j = jax.jacfwd(f_xyzw)(xyzw)
    lz = jnp.sqrt(jnp.sum(j[:, 0:2] * j[:, 0:2]))
    lw = jnp.sqrt(jnp.sum(j[:, 2:4] * j[:, 2:4]))
    return lz, lw


def _complex_loggamma_lanczos(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)
    z1 = z - jnp.complex128(1.0 + 0.0j)
    x = jnp.complex128(_LANCZOS[0] + 0.0j)

    def body(i, acc):
        return acc + _LANCZOS[i] / (z1 + jnp.float64(i))

    x = lax.fori_loop(1, 9, body, x)
    t = z1 + jnp.float64(7.5)
    return el.LOG_SQRT_TWO_PI + (z1 + 0.5) * jnp.log(t) - t + jnp.log(x)


def _complex_loggamma(z: jax.Array) -> jax.Array:
    z = jnp.asarray(z, dtype=jnp.complex128)

    def reflection(w):
        return el.LOG_PI - jnp.log(jnp.sin(el.PI * w)) - _complex_loggamma_lanczos(1.0 - w)

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
    angles = jnp.linspace(0.0, el.TWO_PI, _LIP_SAMPLES, endpoint=False)
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


def _real_partial_bound(fn, x: jax.Array, y: jax.Array, argnum: int) -> jax.Array:
    return jnp.abs(jax.grad(fn, argnums=argnum)(x, y))


def _rigorous_real_bivariate(fn, x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    mid_x, rad_x = _ball_from_interval(x)
    mid_y, rad_y = _ball_from_interval(y)
    val = fn(mid_x, mid_y)

    ts = jnp.linspace(-1.0, 1.0, _LIP_SAMPLES)
    xs = mid_x + rad_x * ts
    ys = mid_y + rad_y * ts
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    pts_x = jnp.ravel(gx)
    pts_y = jnp.ravel(gy)

    lx = jnp.max(jax.vmap(lambda a, b: _real_partial_bound(fn, a, b, 0))(pts_x, pts_y))
    ly = jnp.max(jax.vmap(lambda a, b: _real_partial_bound(fn, a, b, 1))(pts_x, pts_y))
    rad_out = lx * rad_x + ly * rad_y + eps
    out = di.interval(di._below(val - rad_out), di._above(val + rad_out))
    return jnp.where(jnp.isfinite(val), out, _full_interval())


def _adaptive_real_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    mid_x, rad_x = _ball_from_interval(x)
    mid_y, rad_y = _ball_from_interval(y)
    v0 = fn(mid_x, mid_y)

    ts = jnp.linspace(-1.0, 1.0, samples)
    gx, gy = jnp.meshgrid(ts, ts, indexing="ij")
    pts_x = mid_x + rad_x * jnp.ravel(gx)
    pts_y = mid_y + rad_y * jnp.ravel(gy)
    vals = jax.vmap(fn)(pts_x, pts_y)

    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = di.interval(di._below(v0 - rad_out), di._above(v0 + rad_out))
    return jnp.where(jnp.isfinite(v0), out, _full_interval())


def _rigorous_complex_bivariate(fn, x: jax.Array, y: jax.Array, eps: float) -> jax.Array:
    mid_x, rad_x = _ball_from_box(x)
    mid_y, rad_y = _ball_from_box(y)
    val = fn(mid_x, mid_y)

    angles = jnp.linspace(0.0, el.TWO_PI, _LIP_SAMPLES, endpoint=False)
    zs = mid_x + rad_x * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ws = mid_y + rad_y * (jnp.cos(angles) + 1j * jnp.sin(angles))
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    pts_z = jnp.ravel(gz)
    pts_w = jnp.ravel(gw)

    partials = jax.vmap(lambda a, b: _complex_partial_bounds_bivariate(fn, a, b))(pts_z, pts_w)
    lz = jnp.max(partials[0])
    lw = jnp.max(partials[1])
    rad_out = lz * rad_x + lw * rad_y + eps
    finite = jnp.isfinite(jnp.real(val)) & jnp.isfinite(jnp.imag(val))
    out = _box_from_ball(val, rad_out)
    return jnp.where(finite, out, _full_box())


def _adaptive_complex_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    mid_x, rad_x = _ball_from_box(x)
    mid_y, rad_y = _ball_from_box(y)
    v0 = fn(mid_x, mid_y)

    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
    zs = mid_x + rad_x * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ws = mid_y + rad_y * (jnp.cos(angles) + 1j * jnp.sin(angles))
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))

    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    finite = jnp.isfinite(jnp.real(v0)) & jnp.isfinite(jnp.imag(v0))
    out = _box_from_ball(v0, rad_out)
    return jnp.where(finite, out, _full_box())


def _adaptive_complex(fn, x: jax.Array, eps: float, samples: int) -> jax.Array:
    mid, rad = _ball_from_box(x)
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
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


def _interval_hull(a: jax.Array, b: jax.Array) -> jax.Array:
    lo = jnp.minimum(a[0], b[0])
    hi = jnp.maximum(a[1], b[1])
    return di.interval(lo, hi)


def _bivariate_sample_interval(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    x_vals = jnp.asarray([x[0], x[1], 0.5 * (x[0] + x[1])], dtype=jnp.float64)
    y_vals = jnp.asarray([y[0], y[1], 0.5 * (y[0] + y[1])], dtype=jnp.float64)
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(y_vals))(x_vals).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))
    return jnp.where(finite, out, _full_interval())


def _sample_interval_bivariate_grid(fn, x: jax.Array, y: jax.Array, samples: int = 3) -> jax.Array:
    xs = jnp.linspace(x[0], x[1], samples)
    ys = jnp.linspace(y[0], y[1], samples)
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(ys))(xs).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))
    return jnp.where(finite, out, _full_interval())


def _sample_box_bivariate_grid(fn, x: jax.Array, y: jax.Array, samples: int = 3) -> jax.Array:
    xm, xr = _ball_from_box(x)
    ym, yr = _ball_from_box(y)
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
    zs = xm + xr * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ws = ym + yr * (jnp.cos(angles) + 1j * jnp.sin(angles))
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))
    finite = jnp.all(jnp.isfinite(jnp.real(vals)) & jnp.isfinite(jnp.imag(vals)))
    re = di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals))))
    im = di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals))))
    out = acb_core.acb_box(re, im)
    return jnp.where(finite, out, _full_box())


def _sample_box_bivariate_candidates(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    xm, xr = _ball_from_box(x)
    ym, yr = _ball_from_box(y)
    z_offsets = jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j], dtype=el.complex_dtype_from(xm, ym))
    w_offsets = z_offsets
    zs = xm + xr * z_offsets
    ws = ym + yr * w_offsets
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))
    finite = jnp.all(jnp.isfinite(jnp.real(vals)) & jnp.isfinite(jnp.imag(vals)))
    re = di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals))))
    im = di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals))))
    out = acb_core.acb_box(re, im)
    return jnp.where(finite, out, _full_box())


def _bivariate_sample_interval_dense(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    xm = 0.5 * (x[0] + x[1])
    xr = 0.5 * (x[1] - x[0])
    ym = 0.5 * (y[0] + y[1])
    yr = 0.5 * (y[1] - y[0])
    x_vals = jnp.asarray([x[0], x[1], xm, xm - 0.5 * xr, xm + 0.5 * xr], dtype=jnp.float64)
    y_vals = jnp.asarray([y[0], y[1], ym, ym - 0.5 * yr, ym + 0.5 * yr], dtype=jnp.float64)
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(y_vals))(x_vals).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))
    return jnp.where(finite, out, _full_interval())


def _nu_interval_crosses_integer(nu: jax.Array) -> jax.Array:
    n0 = jnp.ceil(nu[0])
    n1 = jnp.floor(nu[1])
    return n0 <= n1


def _nu_box_crosses_integer(nu: jax.Array) -> jax.Array:
    re = acb_core.acb_real(nu)
    im = acb_core.acb_imag(nu)
    return _nu_interval_crosses_integer(re) & (im[0] <= 0.0) & (im[1] >= 0.0)


def _bessel_asym_tail_abs(nu_mid: jax.Array, z_mid_abs: jax.Array) -> jax.Array:
    nu_abs = jnp.abs(nu_mid)
    mu = 4.0 * nu_abs * nu_abs
    t1 = (mu + 1.0) / (8.0 * z_mid_abs)
    t2 = ((mu + 1.0) * (mu + 9.0)) / (128.0 * z_mid_abs * z_mid_abs)
    return t1 + t2


def _bessel_real_asym_interval(kind: str, nu_mid: jax.Array, z_mid: jax.Array, eps: jax.Array) -> jax.Array:
    z_abs = jnp.maximum(jnp.abs(z_mid), jnp.float64(1e-12))
    tail = _bessel_asym_tail_abs(nu_mid, z_abs)
    if kind == "j":
        val = _real_bessel_asym_j(nu_mid, z_mid)
        amp = jnp.sqrt(jnp.float64(2.0) / (el.PI * z_abs))
    elif kind == "y":
        val = _real_bessel_asym_y(nu_mid, z_mid)
        amp = jnp.sqrt(jnp.float64(2.0) / (el.PI * z_abs))
    elif kind == "i":
        val = _real_bessel_asym_i(nu_mid, z_mid)
        amp = jnp.exp(z_mid) / jnp.sqrt(el.TWO_PI * z_abs)
    elif kind == "k":
        val = _real_bessel_asym_k(nu_mid, z_mid)
        amp = jnp.sqrt(el.PI / (jnp.float64(2.0) * z_abs)) * jnp.exp(-z_mid)
    elif kind == "i_scaled":
        val = jnp.exp(-z_mid) * _real_bessel_asym_i(nu_mid, z_mid)
        amp = 1.0 / jnp.sqrt(el.TWO_PI * z_abs)
    elif kind == "k_scaled":
        val = jnp.exp(z_mid) * _real_bessel_asym_k(nu_mid, z_mid)
        amp = jnp.sqrt(el.PI / (jnp.float64(2.0) * z_abs))
    else:
        return _full_interval()
    err = amp * tail + eps
    return di.interval(di._below(val - err), di._above(val + err))


def _hull_with_real_bessel_asym(base: jax.Array, nu: jax.Array, z: jax.Array, kind: str, eps: jax.Array) -> jax.Array:
    nu_mid = 0.5 * (nu[0] + nu[1])
    z_mid = 0.5 * (z[0] + z[1])
    asym = _bessel_real_asym_interval(kind, nu_mid, z_mid, eps)
    use_asym = (z[0] > 12.0) & (z[0] > 0.0)
    return jnp.where(use_asym, _interval_hull(base, asym), base)


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
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
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
    term0 = jnp.power(half, nu) / jnp.exp(lax.lgamma(nu + 1.0))
    sum0 = term0
    z2 = z * z

    def body(k, state):
        term, s = state
        k1 = jnp.float64(k + 1)
        den = k1 * (k1 + nu)
        num = 0.25 * sign * z2
        term = term * (num / den)
        return term, s + term

    _, s = lax.fori_loop(0, 79, body, (term0, sum0))
    return s


def _real_bessel_asym_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.float64(2.0) / (el.PI * z)) * jnp.cos(z - el.HALF_PI * nu - jnp.float64(0.25) * el.PI)


def _real_bessel_asym_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.float64(2.0) / (el.PI * z)) * jnp.sin(z - el.HALF_PI * nu - jnp.float64(0.25) * el.PI)


def _real_bessel_asym_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.exp(z) / jnp.sqrt(el.TWO_PI * z)


def _real_bessel_asym_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.sqrt(el.PI / (jnp.float64(2.0) * z)) * jnp.exp(-z)


def _real_bessel_eval_j(nu: jax.Array, z: jax.Array) -> jax.Array:
    use_asym = (jnp.abs(z) > 12.0) & (z > 0.0)
    return jnp.where(use_asym, _real_bessel_asym_j(nu, z), _real_bessel_series(nu, z, -1.0))


def _real_bessel_eval_i(nu: jax.Array, z: jax.Array) -> jax.Array:
    use_asym = (jnp.abs(z) > 12.0) & (z > 0.0)
    return jnp.where(use_asym, _real_bessel_asym_i(nu, z), _real_bessel_series(nu, z, 1.0))


def _real_bessel_eval_y(nu: jax.Array, z: jax.Array) -> jax.Array:
    use_asym = (jnp.abs(z) > 12.0) & (z > 0.0)
    return jnp.where(use_asym, _real_bessel_asym_y(nu, z), _real_bessel_y(nu, z))


def _real_bessel_eval_k(nu: jax.Array, z: jax.Array) -> jax.Array:
    use_asym = (jnp.abs(z) > 12.0) & (z > 0.0)
    return jnp.where(use_asym, _real_bessel_asym_k(nu, z), _real_bessel_k(nu, z))


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
def arb_ball_sqrt(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.sqrt, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_sin(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.sin, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_cos(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.cos, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_tan(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.tan, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_sinh(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.sinh, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_cosh(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.cosh, x, eps)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_tanh(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _rigorous_real(jnp.tanh, x, eps)


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


def bdg_interval_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(tau)) > 0.0
    rig = _map_interval_bivariate(
        lambda zz, tt: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_log_barnesdoublegamma(a, b, prec_bits)), zz, tt
        ),
        z,
        tau,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_log_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(tau)) > 0.0
    adp = _map_interval_bivariate(
        lambda zz, tt: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_log_barnesdoublegamma(a, b, prec_bits)), zz, tt, samples=_LIP_SAMPLES
        ),
        z,
        tau,
    )
    return jnp.where(valid[..., None], adp, _full_interval())


def bdg_interval_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(tau)) > 0.0
    rig = _map_interval_bivariate(
        lambda zz, tt: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_barnesdoublegamma(a, b, prec_bits)), zz, tt
        ),
        z,
        tau,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(tau)) > 0.0
    adp = _map_interval_bivariate(
        lambda zz, tt: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_barnesdoublegamma(a, b, prec_bits)), zz, tt, samples=_LIP_SAMPLES
        ),
        z,
        tau,
    )
    return jnp.where(valid[..., None], adp, _full_interval())


def bdg_interval_log_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    rig = _map_interval_bivariate(
        lambda ww, bb: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_log_barnesgamma2(a, b, prec_bits)), ww, bb
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_log_barnesgamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    adp = _map_interval_bivariate(
        lambda ww, bb: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_log_barnesgamma2(a, b, prec_bits)), ww, bb, samples=_LIP_SAMPLES
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], adp, _full_interval())


def bdg_interval_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    rig = _map_interval_bivariate(
        lambda ww, bb: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_barnesgamma2(a, b, prec_bits)), ww, bb
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_barnesgamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    adp = _map_interval_bivariate(
        lambda ww, bb: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_barnesgamma2(a, b, prec_bits)), ww, bb, samples=_LIP_SAMPLES
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], adp, _full_interval())


def bdg_interval_log_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    rig = _map_interval_bivariate(
        lambda ww, bb: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_log_normalizeddoublegamma(a, b, prec_bits)), ww, bb
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_log_normalizeddoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    adp = _map_interval_bivariate(
        lambda ww, bb: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_log_normalizeddoublegamma(a, b, prec_bits)), ww, bb, samples=_LIP_SAMPLES
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], adp, _full_interval())


def bdg_interval_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    rig = _map_interval_bivariate(
        lambda ww, bb: _bivariate_sample_interval(
            lambda a, b: jnp.real(double_gamma.bdg_normalizeddoublegamma(a, b, prec_bits)), ww, bb
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], rig, _full_interval())


def bdg_interval_normalizeddoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    valid = di.lower(di.as_interval(beta)) > 0.0
    adp = _map_interval_bivariate(
        lambda ww, bb: _sample_interval_bivariate_grid(
            lambda a, b: jnp.real(double_gamma.bdg_normalizeddoublegamma(a, b, prec_bits)), ww, bb, samples=_LIP_SAMPLES
        ),
        w,
        beta,
    )
    return jnp.where(valid[..., None], adp, _full_interval())

def bdg_complex_log_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, tt: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_log_barnesdoublegamma(a, b, prec_bits), zz, tt),
        z,
        tau,
    )


def bdg_complex_log_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, tt: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_log_barnesdoublegamma(a, b, prec_bits), zz, tt, samples=6),
        z,
        tau,
    )


def bdg_complex_barnesdoublegamma(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, tt: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_barnesdoublegamma(a, b, prec_bits), zz, tt),
        z,
        tau,
    )


def bdg_complex_barnesdoublegamma_adaptive(z: jax.Array, tau: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, tt: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_barnesdoublegamma(a, b, prec_bits), zz, tt, samples=6),
        z,
        tau,
    )


def bdg_complex_log_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_log_barnesgamma2(a, b, prec_bits), ww, bb),
        w,
        beta,
    )


def bdg_complex_log_barnesgamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_log_barnesgamma2(a, b, prec_bits), ww, bb, samples=6),
        w,
        beta,
    )


def bdg_complex_barnesgamma2(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_barnesgamma2(a, b, prec_bits), ww, bb),
        w,
        beta,
    )


def bdg_complex_barnesgamma2_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_barnesgamma2(a, b, prec_bits), ww, bb, samples=6),
        w,
        beta,
    )


def bdg_complex_log_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_log_normalizeddoublegamma(a, b, prec_bits), ww, bb),
        w,
        beta,
    )


def bdg_complex_log_normalizeddoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_log_normalizeddoublegamma(a, b, prec_bits), ww, bb, samples=6),
        w,
        beta,
    )


def bdg_complex_normalizeddoublegamma(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_candidates(lambda a, b: double_gamma.bdg_normalizeddoublegamma(a, b, prec_bits), ww, bb),
        w,
        beta,
    )


def bdg_complex_normalizeddoublegamma_adaptive(w: jax.Array, beta: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda ww, bb: _sample_box_bivariate_grid(lambda a, b: double_gamma.bdg_normalizeddoublegamma(a, b, prec_bits), ww, bb, samples=6),
        w,
        beta,
    )


def bdg_complex_double_sine(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, bb: _sample_box_bivariate_candidates(lambda a, b0: double_gamma.bdg_double_sine(a, b0, prec_bits), zz, bb),
        z,
        b,
    )


def bdg_complex_double_sine_adaptive(z: jax.Array, b: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return _map_box_bivariate(
        lambda zz, bb: _sample_box_bivariate_grid(lambda a, b0: double_gamma.bdg_double_sine(a, b0, prec_bits), zz, bb, samples=6),
        z,
        b,
    )


def arb_ball_gamma(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(_gamma_real, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_lgamma(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lax.lgamma, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_rgamma(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: jnp.exp(-lax.lgamma(v)), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_pow(x: jax.Array, y: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(a, b):
        return jnp.power(a, b)

    def kernel(a, b):
        valid = (di.lower(a) > 0.0) & jnp.isfinite(di.lower(a)) & jnp.isfinite(di.upper(a))
        rig = _rigorous_real_bivariate(fn, a, b, eps)
        fallback = arb_core.arb_pow_prec(a, b, prec_bits)
        safe = _interval_hull(rig, fallback)
        return jnp.where(valid[..., None], safe, fallback)

    return _map_interval_bivariate(kernel, x, y)


@partial(jax.jit, static_argnames=("n", "prec_bits"))
def arb_ball_pow_ui(x: jax.Array, n: int, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(v):
        return jnp.power(v, jnp.float64(n))

    return _map_interval(lambda t: _rigorous_real(fn, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_pow_fmpq(x: jax.Array, p: jax.Array, q: jax.Array, prec_bits: int = 53) -> jax.Array:
    pp = jnp.asarray(p, dtype=jnp.float64)
    qq = jnp.asarray(q, dtype=jnp.float64)
    y = di.interval(pp / qq, pp / qq)
    return arb_ball_pow(x, y, prec_bits=prec_bits)


@partial(jax.jit, static_argnames=("k", "prec_bits"))
def arb_ball_root_ui(x: jax.Array, k: int, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    kk = jnp.float64(k)
    odd = (k % 2) == 1

    def odd_fn(v):
        return jnp.sign(v) * jnp.power(jnp.abs(v), 1.0 / kk)

    def even_fn(v):
        return jnp.power(v, 1.0 / kk)

    def kernel(t):
        if odd:
            return _rigorous_real(odd_fn, t, eps)
        valid = di.lower(t) > 0.0
        rig = _rigorous_real(even_fn, t, eps)
        fallback = arb_core.arb_root_ui_prec(t, k, prec_bits)
        return jnp.where(valid[..., None], rig, fallback)

    return _map_interval(kernel, x)


@partial(jax.jit, static_argnames=("k", "prec_bits"))
def arb_ball_root(x: jax.Array, k: int, prec_bits: int = 53) -> jax.Array:
    return arb_ball_root_ui(x, k, prec_bits=prec_bits)


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
def arb_ball_sqrt_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.sqrt, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_sin_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.sin, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_cos_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.cos, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_tan_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.tan, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_sinh_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.sinh, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_cosh_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.cosh, x, eps, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_tanh_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _adaptive_real(jnp.tanh, x, eps, samples)

@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_gamma_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(_gamma_real, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_lgamma_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lax.lgamma, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_rgamma_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jnp.exp(-lax.lgamma(v)), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_pow_adaptive(x: jax.Array, y: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(a, b):
        return jnp.power(a, b)

    def kernel(a, b):
        valid = (di.lower(a) > 0.0) & jnp.isfinite(di.lower(a)) & jnp.isfinite(di.upper(a))
        adapt = _adaptive_real_bivariate(fn, a, b, eps, samples)
        fallback = arb_core.arb_pow_prec(a, b, prec_bits)
        safe = _interval_hull(adapt, fallback)
        return jnp.where(valid[..., None], safe, fallback)

    return _map_interval_bivariate(kernel, x, y)


@partial(jax.jit, static_argnames=("n", "prec_bits", "samples"))
def arb_ball_pow_ui_adaptive(x: jax.Array, n: int, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: jnp.power(v, jnp.float64(n)), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_pow_fmpq_adaptive(x: jax.Array, p: jax.Array, q: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    pp = jnp.asarray(p, dtype=jnp.float64)
    qq = jnp.asarray(q, dtype=jnp.float64)
    y = di.interval(pp / qq, pp / qq)
    return arb_ball_pow_adaptive(x, y, prec_bits=prec_bits, samples=samples)


@partial(jax.jit, static_argnames=("k", "prec_bits", "samples"))
def arb_ball_root_ui_adaptive(x: jax.Array, k: int, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    kk = jnp.float64(k)
    odd = (k % 2) == 1

    def odd_fn(v):
        return jnp.sign(v) * jnp.power(jnp.abs(v), 1.0 / kk)

    def even_fn(v):
        return jnp.power(v, 1.0 / kk)

    def kernel(t):
        if odd:
            return _adaptive_real(odd_fn, t, eps, samples)
        valid = di.lower(t) > 0.0
        adapt = _adaptive_real(even_fn, t, eps, samples)
        fallback = arb_core.arb_root_ui_prec(t, k, prec_bits)
        return jnp.where(valid[..., None], adapt, fallback)

    return _map_interval(kernel, x)


@partial(jax.jit, static_argnames=("k", "prec_bits", "samples"))
def arb_ball_root_adaptive(x: jax.Array, k: int, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    return arb_ball_root_ui_adaptive(x, k, prec_bits=prec_bits, samples=samples)


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
    return _map_interval(lambda t: _rigorous_real(lax.erf, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfc(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: 1.0 - lax.erf(v), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    erfi_fn = _erfi_real
    return _map_interval(lambda t: _rigorous_real(erfi_fn, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfinv(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(_erfinv_real, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_erfcinv(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: _erfinv_real(1.0 - v), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erf(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(_erf_complex, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erfc(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(lambda z: 1.0 - _erf_complex(z), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_erfi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _rigorous_complex(lambda z: -1j * _erf_complex(1j * z), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_ei(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(_expi_real, t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_si(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: _si_ci_real(v)[0], t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_ci(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: _si_ci_real(v)[1], t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_shi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: 0.5 * (_expi_real(v) - _expi_real(-v)), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_chi(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: 0.5 * (_expi_real(v) + _expi_real(-v)), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_li(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        out = _rigorous_real(lambda v: _expi_real(jnp.log(v)), t, eps)
        full = _full_interval()
        return jnp.where(t[0] <= 0.0, full, out)

    return _map_interval(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_dilog(x: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _rigorous_real(lambda v: _dilog_real(v), t, eps), x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_fresnel(x: jax.Array, prec_bits: int = 53, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        s = _rigorous_real(lambda v: _fresnel_real(v, True)[0], t, eps)
        c = _rigorous_real(lambda v: _fresnel_real(v, True)[1], t, eps)
        if not normalized:
            s = di.fast_mul(s, di.interval(jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0))))
            c = di.fast_mul(c, di.interval(jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0))))
        return s, c

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_airy(x: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        ai = _rigorous_real(lambda v: _airy_real(v)[0], t, eps)
        aip = _rigorous_real(lambda v: _airy_real(v)[1], t, eps)
        bi = _rigorous_real(lambda v: _airy_real(v)[2], t, eps)
        bip = _rigorous_real(lambda v: _airy_real(v)[3], t, eps)
        return ai, aip, bi, bip

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erf_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lax.erf, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfc_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: 1.0 - lax.erf(v), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    erfi_fn = _erfi_real
    return _map_interval(lambda t: _adaptive_real(erfi_fn, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfinv_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(_erfinv_real, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_erfcinv_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: _erfinv_real(1.0 - v), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erf_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(_erf_complex, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erfc_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(lambda z: 1.0 - _erf_complex(z), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_erfi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_box(lambda t: _adaptive_complex(lambda z: -1j * _erf_complex(1j * z), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_ei_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(_expi_real, t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_si_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: _si_ci_real(v)[0], t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_ci_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: _si_ci_real(v)[1], t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_shi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: 0.5 * (_expi_real(v) - _expi_real(-v)), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_chi_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: 0.5 * (_expi_real(v) + _expi_real(-v)), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_li_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        out = _adaptive_real(lambda v: _expi_real(jnp.log(v)), t, eps, samples)
        full = _full_interval()
        return jnp.where(t[0] <= 0.0, full, out)

    return _map_interval(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_dilog_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval(lambda t: _adaptive_real(lambda v: _dilog_real(v), t, eps, samples), x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_fresnel_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9, normalized: bool = False) -> tuple[jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        s = _adaptive_real(lambda v: _fresnel_real(v, True)[0], t, eps, samples)
        c = _adaptive_real(lambda v: _fresnel_real(v, True)[1], t, eps, samples)
        if not normalized:
            s = di.fast_mul(s, di.interval(jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0))))
            c = di.fast_mul(c, di.interval(jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0)), jnp.float64(jnp.sqrt(el.PI) / jnp.sqrt(2.0))))
        return s, c

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_airy_adaptive(x: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(t):
        ai = _adaptive_real(lambda v: _airy_real(v)[0], t, eps, samples)
        aip = _adaptive_real(lambda v: _airy_real(v)[1], t, eps, samples)
        bi = _adaptive_real(lambda v: _airy_real(v)[2], t, eps, samples)
        bip = _adaptive_real(lambda v: _airy_real(v)[3], t, eps, samples)
        return ai, aip, bi, bip

    return _map_interval_pair(fn, x)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_j(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    def bound(a, b):
        base = _rigorous_real_bivariate(lambda u, v: _real_bessel_eval_j(u, v), a, b, eps)
        samp = _bivariate_sample_interval_dense(lambda u, v: _real_bessel_eval_j(u, v), a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "j", eps)
    return _map_interval_bivariate(bound, nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_i(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    def bound(a, b):
        base = _rigorous_real_bivariate(lambda u, v: _real_bessel_eval_i(u, v), a, b, eps)
        samp = _bivariate_sample_interval_dense(lambda u, v: _real_bessel_eval_i(u, v), a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "i", eps)
    return _map_interval_bivariate(bound, nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_y(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        jnu = _real_bessel_eval_j(u, v)
        jneg = _real_bessel_eval_j(-u, v)
        val = (jnu * jnp.cos(el.PI * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    def bound(a, b):
        base = _rigorous_real_bivariate(fn, a, b, eps)
        samp = _bivariate_sample_interval_dense(fn, a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "y", eps)
    return _map_interval_bivariate(
        lambda a, b: jnp.where(_nu_interval_crosses_integer(a), _full_interval(), bound(a, b)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_k(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _real_bessel_eval_i(u, v)
        ineg = _real_bessel_eval_i(-u, v)
        val = el.HALF_PI * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    def bound(a, b):
        base = _rigorous_real_bivariate(fn, a, b, eps)
        samp = _bivariate_sample_interval_dense(fn, a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "k", eps)
    return _map_interval_bivariate(
        lambda a, b: jnp.where(_nu_interval_crosses_integer(a), _full_interval(), bound(a, b)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_jy(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array]:
    return arb_ball_bessel_j(nu, z, prec_bits), arb_ball_bessel_y(nu, z, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_i_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _real_bessel_eval_i(u, v)

    def bound(a, b):
        base = _rigorous_real_bivariate(fn, a, b, eps)
        samp = _bivariate_sample_interval_dense(fn, a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "i_scaled", eps)
    return _map_interval_bivariate(bound, nu, z)


@partial(jax.jit, static_argnames=("prec_bits",))
def arb_ball_bessel_k_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _real_bessel_eval_i(u, v)
        ineg = _real_bessel_eval_i(-u, v)
        k = el.HALF_PI * (ineg - inu) / s
        return jnp.exp(v) * k

    def bound(a, b):
        base = _rigorous_real_bivariate(fn, a, b, eps)
        samp = _bivariate_sample_interval(fn, a, b)
        return _hull_with_real_bessel_asym(_interval_hull(base, samp), a, b, "k_scaled", eps)
    return _map_interval_bivariate(
        lambda a, b: jnp.where(_nu_interval_crosses_integer(a), _full_interval(), bound(a, b)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_j_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(
        lambda a, b: _hull_with_real_bessel_asym(
            _adaptive_real_bivariate(lambda u, v: _real_bessel_eval_j(u, v), a, b, eps, max(samples, 15)),
            a,
            b,
            "j",
            eps,
        ),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_i_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))
    return _map_interval_bivariate(
        lambda a, b: _hull_with_real_bessel_asym(
            _adaptive_real_bivariate(lambda u, v: _real_bessel_eval_i(u, v), a, b, eps, max(samples, 15)),
            a,
            b,
            "i",
            eps,
        ),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_y_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        jnu = _real_bessel_eval_j(u, v)
        jneg = _real_bessel_eval_j(-u, v)
        val = (jnu * jnp.cos(el.PI * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(
        lambda a, b: jnp.where(
            _nu_interval_crosses_integer(a),
            _full_interval(),
            _hull_with_real_bessel_asym(
                _adaptive_real_bivariate(fn, a, b, eps, max(samples, 15)),
                a,
                b,
                "y",
                eps,
            ),
        ),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_k_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _real_bessel_eval_i(u, v)
        ineg = _real_bessel_eval_i(-u, v)
        val = 0.5 * el.PI * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan, val)

    return _map_interval_bivariate(
        lambda a, b: jnp.where(
            _nu_interval_crosses_integer(a),
            _full_interval(),
            _hull_with_real_bessel_asym(
                _adaptive_real_bivariate(fn, a, b, eps, max(samples, 15)),
                a,
                b,
                "k",
                eps,
            ),
        ),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_jy_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array]:
    return arb_ball_bessel_j_adaptive(nu, z, prec_bits, samples), arb_ball_bessel_y_adaptive(nu, z, prec_bits, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_i_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _real_bessel_eval_i(u, v)

    return _map_interval_bivariate(
        lambda a, b: _hull_with_real_bessel_asym(
            _adaptive_real_bivariate(fn, a, b, eps, max(samples, 15)),
            a,
            b,
            "i_scaled",
            eps,
        ),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def arb_ball_bessel_k_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _real_bessel_eval_i(u, v)
        ineg = _real_bessel_eval_i(-u, v)
        k = 0.5 * el.PI * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_interval_bivariate(
        lambda a, b: jnp.where(
            _nu_interval_crosses_integer(a),
            _full_interval(),
            _hull_with_real_bessel_asym(
                _adaptive_real_bivariate(fn, a, b, eps, max(samples, 15)),
                a,
                b,
                "k_scaled",
                eps,
            ),
        ),
        nu,
        z,
    )


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
        s = jnp.sin(el.PI * u)
        jnu = _complex_bessel_series(u, v, -1.0)
        jneg = _complex_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(el.PI * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _rigorous_complex_bivariate(fn, a, b, eps)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_k(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        val = 0.5 * el.PI * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _rigorous_complex_bivariate(fn, a, b, eps)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_jy(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> tuple[jax.Array, jax.Array]:
    return acb_ball_bessel_j(nu, z, prec_bits), acb_ball_bessel_y(nu, z, prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_i_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _complex_bessel_series(u, v, 1.0)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _rigorous_complex_bivariate(fn, a, b, eps)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_ball_bessel_k_scaled(nu: jax.Array, z: jax.Array, prec_bits: int = 53) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        k = 0.5 * el.PI * (ineg - inu) / s
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
        s = jnp.sin(el.PI * u)
        jnu = _complex_bessel_series(u, v, -1.0)
        jneg = _complex_bessel_series(-u, v, -1.0)
        val = (jnu * jnp.cos(el.PI * u) - jneg) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _adaptive_complex_bivariate(fn, a, b, eps, samples)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_k_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        val = 0.5 * el.PI * (ineg - inu) / s
        return jnp.where(jnp.abs(s) < 1e-8, jnp.nan + 1j * jnp.nan, val)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _adaptive_complex_bivariate(fn, a, b, eps, samples)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_jy_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> tuple[jax.Array, jax.Array]:
    return acb_ball_bessel_j_adaptive(nu, z, prec_bits, samples), acb_ball_bessel_y_adaptive(nu, z, prec_bits, samples)


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_i_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        return jnp.exp(-v) * _complex_bessel_series(u, v, 1.0)

    return _map_box_bivariate(
        lambda a, b: jnp.where(_nu_box_crosses_integer(a), _full_box(), _adaptive_complex_bivariate(fn, a, b, eps, samples)),
        nu,
        z,
    )


@partial(jax.jit, static_argnames=("prec_bits", "samples"))
def acb_ball_bessel_k_scaled_adaptive(nu: jax.Array, z: jax.Array, prec_bits: int = 53, samples: int = 9) -> jax.Array:
    eps = jnp.exp2(-jnp.float64(prec_bits))

    def fn(u, v):
        s = jnp.sin(el.PI * u)
        inu = _complex_bessel_series(u, v, 1.0)
        ineg = _complex_bessel_series(-u, v, 1.0)
        k = 0.5 * el.PI * (ineg - inu) / s
        return jnp.exp(v) * k

    return _map_box_bivariate(lambda a, b: _adaptive_complex_bivariate(fn, a, b, eps, samples), nu, z)


def arb_ball_exp_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_exp(x, prec_bits=prec_bits)


def arb_ball_log_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_log(x, prec_bits=prec_bits)


def arb_ball_sqrt_mp(x: jax.Array, dps: int | None = None) -> jax.Array:
    prec_bits = precision.dps_to_bits(precision.get_dps() if dps is None else dps)
    return arb_ball_sqrt(x, prec_bits=prec_bits)


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
    "arb_ball_sqrt",
    "arb_ball_sin",
    "arb_ball_cos",
    "arb_ball_tan",
    "arb_ball_sinh",
    "arb_ball_cosh",
    "arb_ball_tanh",
    "arb_ball_gamma",
    "arb_ball_lgamma",
    "arb_ball_rgamma",
    "arb_ball_pow",
    "arb_ball_pow_ui",
    "arb_ball_pow_fmpq",
    "arb_ball_root_ui",
    "arb_ball_root",
    "acb_ball_exp",
    "acb_ball_log",
    "acb_ball_sin",
    "acb_ball_gamma",
    "arb_ball_exp_adaptive",
    "arb_ball_log_adaptive",
    "arb_ball_sqrt_adaptive",
    "arb_ball_sin_adaptive",
    "arb_ball_cos_adaptive",
    "arb_ball_tan_adaptive",
    "arb_ball_sinh_adaptive",
    "arb_ball_cosh_adaptive",
    "arb_ball_tanh_adaptive",
    "arb_ball_gamma_adaptive",
    "arb_ball_lgamma_adaptive",
    "arb_ball_rgamma_adaptive",
    "arb_ball_pow_adaptive",
    "arb_ball_pow_ui_adaptive",
    "arb_ball_pow_fmpq_adaptive",
    "arb_ball_root_ui_adaptive",
    "arb_ball_root_adaptive",
    "acb_ball_exp_adaptive",
    "acb_ball_log_adaptive",
    "acb_ball_sin_adaptive",
    "acb_ball_gamma_adaptive",
    "arb_ball_exp_mp",
    "arb_ball_log_mp",
    "arb_ball_sqrt_mp",
    "arb_ball_sin_mp",
    "arb_ball_gamma_mp",
    "acb_ball_exp_mp",
    "acb_ball_log_mp",
    "acb_ball_sin_mp",
    "acb_ball_gamma_mp",
]
