from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def _as_complex_vector(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.complex128)
    if arr.ndim != 1:
        raise ValueError("expected a 1D complex vector")
    return arr


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _dft_matrix(n: int, inverse: bool = False) -> jax.Array:
    k = jnp.arange(n, dtype=jnp.float64)[:, None]
    t = jnp.arange(n, dtype=jnp.float64)[None, :]
    sign = 1.0 if inverse else -1.0
    expo = sign * 2.0j * jnp.pi * (k * t) / float(n)
    return jnp.exp(expo)


def as_acb_box(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.shape[-1] != 4:
        raise ValueError("expected acb interval boxes with last dimension 4")
    return arr


def acb_box(real_interval: jax.Array, imag_interval: jax.Array) -> jax.Array:
    r = di.as_interval(real_interval)
    i = di.as_interval(imag_interval)
    return jnp.concatenate([r, i], axis=-1)


def acb_real(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    return xb[..., 0:2]


def acb_imag(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    return xb[..., 2:4]


def _acb_add(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    return acb_box(di.fast_add(acb_real(xb), acb_real(yb)), di.fast_add(acb_imag(xb), acb_imag(yb)))


def _acb_mul(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    xr, xi = acb_real(xb), acb_imag(xb)
    yr, yi = acb_real(yb), acb_imag(yb)
    ac = di.fast_mul(xr, yr)
    bd = di.fast_mul(xi, yi)
    ad = di.fast_mul(xr, yi)
    bc = di.fast_mul(xi, yr)
    return acb_box(di.fast_sub(ac, bd), di.fast_add(ad, bc))


def _acb_scale_real(x: jax.Array, s: float) -> jax.Array:
    xb = as_acb_box(x)
    si = di.interval(jnp.float64(s), jnp.float64(s))
    return acb_box(di.fast_mul(acb_real(xb), si), di.fast_mul(acb_imag(xb), si))


def _acb_zero_like_len(n: int) -> jax.Array:
    z = jnp.zeros((n,), dtype=jnp.float64)
    return acb_box(di.interval(z, z), di.interval(z, z))


def _acb_twiddle_matrix(n: int, inverse: bool = False) -> jax.Array:
    k = jnp.arange(n, dtype=jnp.float64)[:, None]
    t = jnp.arange(n, dtype=jnp.float64)[None, :]
    sign = 1.0 if inverse else -1.0
    ang = sign * 2.0 * jnp.pi * (k * t) / float(n)
    wr = jnp.cos(ang)
    wi = jnp.sin(ang)
    return jnp.stack([wr, wr, wi, wi], axis=-1)


@partial(jax.jit, static_argnames=("inverse",))
def dft_naive(x: jax.Array, inverse: bool = False) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    mat = _dft_matrix(n, inverse=inverse)
    y = mat @ x
    if inverse:
        y = y / float(n)
    return y


@jax.jit
def idft_naive(x: jax.Array) -> jax.Array:
    return dft_naive(x, inverse=True)


@jax.jit
def dft_rad2(x: jax.Array) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    if _is_power_of_two(n):
        return jnp.fft.fft(x)
    return dft_naive(x)


@jax.jit
def idft_rad2(x: jax.Array) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    if _is_power_of_two(n):
        return jnp.fft.ifft(x)
    return idft_naive(x)


@jax.jit
def dft(x: jax.Array) -> jax.Array:
    return dft_rad2(x)


@jax.jit
def idft(x: jax.Array) -> jax.Array:
    return idft_rad2(x)


@partial(jax.jit, static_argnames=("cyc",))
def dft_prod(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    x = _as_complex_vector(x)
    n = 1
    for m in cyc:
        n *= int(m)
    if n != x.shape[0]:
        raise ValueError("product(cyc) must equal len(x)")
    return dft(x)


@jax.jit
def convol_circular_naive(f: jax.Array, g: jax.Array) -> jax.Array:
    f = _as_complex_vector(f)
    g = _as_complex_vector(g)
    if f.shape[0] != g.shape[0]:
        raise ValueError("f and g must have the same length")
    n = f.shape[0]
    k = jnp.arange(n, dtype=jnp.int64)[:, None]
    y = jnp.arange(n, dtype=jnp.int64)[None, :]
    idx = (k - y) % n
    ff = f[idx]
    return jnp.sum(ff * g[None, :], axis=1)


@jax.jit
def convol_circular_dft(f: jax.Array, g: jax.Array) -> jax.Array:
    f = _as_complex_vector(f)
    g = _as_complex_vector(g)
    if f.shape[0] != g.shape[0]:
        raise ValueError("f and g must have the same length")
    return idft(dft(f) * dft(g))


@jax.jit
def convol_circular_rad2(f: jax.Array, g: jax.Array) -> jax.Array:
    f = _as_complex_vector(f)
    g = _as_complex_vector(g)
    if f.shape[0] != g.shape[0]:
        raise ValueError("f and g must have the same length")
    return idft_rad2(dft_rad2(f) * dft_rad2(g))


@jax.jit
def convol_circular(f: jax.Array, g: jax.Array) -> jax.Array:
    return convol_circular_rad2(f, g)


@partial(jax.jit, static_argnames=("inverse",))
def acb_dft_naive(x: jax.Array, inverse: bool = False) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    w = _acb_twiddle_matrix(n, inverse=inverse)

    def row_accum(wrow):
        acc0 = acb_box(di.interval(jnp.float64(0.0), jnp.float64(0.0)), di.interval(jnp.float64(0.0), jnp.float64(0.0)))

        def body(i, acc):
            xi = xb[i]
            wi = wrow[i]
            return _acb_add(acc, _acb_mul(xi, wi))

        out = lax.fori_loop(0, n, body, acc0)
        return out

    y = jax.vmap(row_accum)(w)
    if inverse:
        y = _acb_scale_real(y, 1.0 / float(n))
    return y


@jax.jit
def acb_idft_naive(x: jax.Array) -> jax.Array:
    return acb_dft_naive(x, inverse=True)


@jax.jit
def acb_dft_rad2(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_naive(xb, inverse=False)
    return acb_dft_naive(xb, inverse=False)


@jax.jit
def acb_idft_rad2(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_naive(xb, inverse=True)
    return acb_dft_naive(xb, inverse=True)


@jax.jit
def acb_dft(x: jax.Array) -> jax.Array:
    return acb_dft_rad2(x)


@jax.jit
def acb_idft(x: jax.Array) -> jax.Array:
    return acb_idft_rad2(x)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_prod(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    xb = as_acb_box(x)
    n = 1
    for m in cyc:
        n *= int(m)
    if n != xb.shape[0]:
        raise ValueError("product(cyc) must equal len(x)")
    return acb_dft(xb)


@jax.jit
def acb_convol_circular_naive(f: jax.Array, g: jax.Array) -> jax.Array:
    fb = as_acb_box(f)
    gb = as_acb_box(g)
    if fb.shape[0] != gb.shape[0]:
        raise ValueError("f and g must have the same length")
    n = fb.shape[0]
    out0 = _acb_zero_like_len(n)

    def body_x(ix, out):
        acc0 = acb_box(di.interval(jnp.float64(0.0), jnp.float64(0.0)), di.interval(jnp.float64(0.0), jnp.float64(0.0)))

        def body_y(iy, acc):
            idx = (ix + n - iy) % n
            fx = fb[idx]
            gy = gb[iy]
            return _acb_add(acc, _acb_mul(fx, gy))

        val = lax.fori_loop(0, n, body_y, acc0)
        return out.at[ix, :].set(val)

    return lax.fori_loop(0, n, body_x, out0)


@jax.jit
def acb_convol_circular_dft(f: jax.Array, g: jax.Array) -> jax.Array:
    fb = as_acb_box(f)
    gb = as_acb_box(g)
    if fb.shape[0] != gb.shape[0]:
        raise ValueError("f and g must have the same length")
    return acb_idft(acb_mul_vec(acb_dft(fb), acb_dft(gb)))


def acb_mul_vec(x: jax.Array, y: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    yb = as_acb_box(y)
    xr, xi = acb_real(xb), acb_imag(xb)
    yr, yi = acb_real(yb), acb_imag(yb)
    ac = di.fast_mul(xr, yr)
    bd = di.fast_mul(xi, yi)
    ad = di.fast_mul(xr, yi)
    bc = di.fast_mul(xi, yr)
    return acb_box(di.fast_sub(ac, bd), di.fast_add(ad, bc))


@jax.jit
def acb_convol_circular_rad2(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_idft_rad2(acb_mul_vec(acb_dft_rad2(f), acb_dft_rad2(g)))


@jax.jit
def acb_convol_circular(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_rad2(f, g)


dft_naive_jit = jax.jit(dft_naive, static_argnames=("inverse",))
idft_naive_jit = jax.jit(idft_naive)
dft_rad2_jit = jax.jit(dft_rad2)
idft_rad2_jit = jax.jit(idft_rad2)
dft_jit = jax.jit(dft)
idft_jit = jax.jit(idft)
dft_prod_jit = jax.jit(dft_prod, static_argnames=("cyc",))
convol_circular_naive_jit = jax.jit(convol_circular_naive)
convol_circular_dft_jit = jax.jit(convol_circular_dft)
convol_circular_rad2_jit = jax.jit(convol_circular_rad2)
convol_circular_jit = jax.jit(convol_circular)

acb_dft_naive_jit = jax.jit(acb_dft_naive, static_argnames=("inverse",))
acb_idft_naive_jit = jax.jit(acb_idft_naive)
acb_dft_rad2_jit = jax.jit(acb_dft_rad2)
acb_idft_rad2_jit = jax.jit(acb_idft_rad2)
acb_dft_jit = jax.jit(acb_dft)
acb_idft_jit = jax.jit(acb_idft)
acb_dft_prod_jit = jax.jit(acb_dft_prod, static_argnames=("cyc",))
acb_convol_circular_naive_jit = jax.jit(acb_convol_circular_naive)
acb_convol_circular_dft_jit = jax.jit(acb_convol_circular_dft)
acb_convol_circular_rad2_jit = jax.jit(acb_convol_circular_rad2)
acb_convol_circular_jit = jax.jit(acb_convol_circular)


__all__ = [
    "dft_naive",
    "idft_naive",
    "dft_rad2",
    "idft_rad2",
    "dft",
    "idft",
    "dft_prod",
    "convol_circular_naive",
    "convol_circular_dft",
    "convol_circular_rad2",
    "convol_circular",
    "dft_naive_jit",
    "idft_naive_jit",
    "dft_rad2_jit",
    "idft_rad2_jit",
    "dft_jit",
    "idft_jit",
    "dft_prod_jit",
    "convol_circular_naive_jit",
    "convol_circular_dft_jit",
    "convol_circular_rad2_jit",
    "convol_circular_jit",
    "as_acb_box",
    "acb_box",
    "acb_real",
    "acb_imag",
    "acb_dft_naive",
    "acb_idft_naive",
    "acb_dft_rad2",
    "acb_idft_rad2",
    "acb_dft",
    "acb_idft",
    "acb_dft_prod",
    "acb_convol_circular_naive",
    "acb_convol_circular_dft",
    "acb_convol_circular_rad2",
    "acb_convol_circular",
    "acb_dft_naive_jit",
    "acb_idft_naive_jit",
    "acb_dft_rad2_jit",
    "acb_idft_rad2_jit",
    "acb_dft_jit",
    "acb_idft_jit",
    "acb_dft_prod_jit",
    "acb_convol_circular_naive_jit",
    "acb_convol_circular_dft_jit",
    "acb_convol_circular_rad2_jit",
    "acb_convol_circular_jit",
]
