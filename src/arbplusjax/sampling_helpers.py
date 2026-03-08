from __future__ import annotations

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import elementary as el


def full_interval() -> jax.Array:
    return di.interval(-jnp.inf, jnp.inf)


def full_box() -> jax.Array:
    return jnp.array([-jnp.inf, jnp.inf, -jnp.inf, jnp.inf], dtype=jnp.float64)


def ball_from_interval(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = di.as_interval(x)
    mid = di.midpoint(x)
    rad = 0.5 * (x[1] - x[0])
    return mid, jnp.maximum(rad, 0.0)


def box_from_ball(mid: jax.Array, rad: jax.Array) -> jax.Array:
    rad = jnp.maximum(rad, 0.0)
    return jnp.array(
        [jnp.real(mid) - rad, jnp.real(mid) + rad, jnp.imag(mid) - rad, jnp.imag(mid) + rad],
        dtype=jnp.float64,
    )


def ball_from_box(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    box = acb_core.as_acb_box(x)
    re = acb_core.acb_real(box)
    im = acb_core.acb_imag(box)
    mid = di.midpoint(re) + 1j * di.midpoint(im)
    rad = jnp.maximum(0.5 * (re[1] - re[0]), 0.5 * (im[1] - im[0]))
    return mid, rad


def adaptive_real(fn, x: jax.Array, eps: float, samples: int) -> jax.Array:
    mid, rad = ball_from_interval(x)
    t = jnp.linspace(-1.0, 1.0, samples)
    xs = mid + rad * t
    vals = jax.vmap(fn)(xs)
    v0 = fn(mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = di.interval(di._below(v0 - rad_out), di._above(v0 + rad_out))
    return jnp.where(jnp.isfinite(v0), out, full_interval())


def adaptive_real_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    mid_x, rad_x = ball_from_interval(x)
    mid_y, rad_y = ball_from_interval(y)
    ts = jnp.linspace(-1.0, 1.0, samples)
    gx, gy = jnp.meshgrid(ts, ts, indexing="ij")
    pts_x = mid_x + rad_x * jnp.ravel(gx)
    pts_y = mid_y + rad_y * jnp.ravel(gy)
    vals = jax.vmap(fn)(pts_x, pts_y)
    v0 = fn(mid_x, mid_y)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = di.interval(di._below(v0 - rad_out), di._above(v0 + rad_out))
    return jnp.where(jnp.isfinite(v0), out, full_interval())


def adaptive_complex(fn, x: jax.Array, eps: float, samples: int) -> jax.Array:
    mid, rad = ball_from_box(x)
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
    zs = mid + rad * (jnp.cos(angles) + 1j * jnp.sin(angles))
    vals = jax.vmap(fn)(zs)
    v0 = fn(mid)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    finite = jnp.isfinite(jnp.real(v0)) & jnp.isfinite(jnp.imag(v0))
    out = box_from_ball(v0, rad_out)
    return jnp.where(finite, out, full_box())


def adaptive_complex_bivariate(fn, x: jax.Array, y: jax.Array, eps: float, samples: int) -> jax.Array:
    mid_x, rad_x = ball_from_box(x)
    mid_y, rad_y = ball_from_box(y)
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
    zs = mid_x + rad_x * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ws = mid_y + rad_y * (jnp.cos(angles) + 1j * jnp.sin(angles))
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))
    v0 = fn(mid_x, mid_y)
    rad_out = jnp.max(jnp.abs(vals - v0)) + eps
    out = box_from_ball(v0, rad_out)
    finite = jnp.isfinite(jnp.real(v0)) & jnp.isfinite(jnp.imag(v0))
    return jnp.where(finite, out, full_box())


def sample_interval_bivariate_grid(fn, x: jax.Array, y: jax.Array, samples: int = 3) -> jax.Array:
    xs = jnp.linspace(x[0], x[1], samples)
    ys = jnp.linspace(y[0], y[1], samples)
    vals = jax.vmap(lambda a: jax.vmap(lambda b: fn(a, b))(ys))(xs).reshape(-1)
    finite = jnp.all(jnp.isfinite(vals))
    out = di.interval(di._below(jnp.min(vals)), di._above(jnp.max(vals)))
    return jnp.where(finite, out, full_interval())


def sample_box_bivariate_grid(fn, x: jax.Array, y: jax.Array, samples: int = 3) -> jax.Array:
    xm, xr = ball_from_box(x)
    ym, yr = ball_from_box(y)
    angles = jnp.linspace(0.0, el.TWO_PI, samples, endpoint=False)
    zs = xm + xr * (jnp.cos(angles) + 1j * jnp.sin(angles))
    ws = ym + yr * (jnp.cos(angles) + 1j * jnp.sin(angles))
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))
    finite = jnp.all(jnp.isfinite(jnp.real(vals)) & jnp.isfinite(jnp.imag(vals)))
    re = di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals))))
    im = di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals))))
    out = acb_core.acb_box(re, im)
    return jnp.where(finite, out, full_box())


def sample_box_bivariate_candidates(fn, x: jax.Array, y: jax.Array) -> jax.Array:
    xm, xr = ball_from_box(x)
    ym, yr = ball_from_box(y)
    c_dtype = el.complex_dtype_from(xm, ym)
    z_offsets = jnp.asarray([0.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j], dtype=c_dtype)
    zs = xm + xr * z_offsets
    ws = ym + yr * z_offsets
    gz, gw = jnp.meshgrid(zs, ws, indexing="ij")
    vals = jax.vmap(fn)(jnp.ravel(gz), jnp.ravel(gw))
    finite = jnp.all(jnp.isfinite(jnp.real(vals)) & jnp.isfinite(jnp.imag(vals)))
    re = di.interval(di._below(jnp.min(jnp.real(vals))), di._above(jnp.max(jnp.real(vals))))
    im = di.interval(di._below(jnp.min(jnp.imag(vals))), di._above(jnp.max(jnp.imag(vals))))
    out = acb_core.acb_box(re, im)
    return jnp.where(finite, out, full_box())
