from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import elementary as el
from . import checks
from . import kernel_helpers as kh
from . import transform_common as tc
from . import wrappers_common as wc


def _as_complex_vector(x: jax.Array) -> jax.Array:
    return tc.as_complex_vector(x, "dft._as_complex_vector")


def _as_complex_array(x: jax.Array) -> jax.Array:
    return tc.as_complex_array(x, "dft._as_complex_array")


def _is_power_of_two(n: int) -> bool:
    return tc.is_power_of_two(n)


def _canonical_axes(ndim: int, axes: tuple[int, ...] | None) -> tuple[int, ...]:
    return tc.canonical_axes(ndim, axes, "dft._canonical_axes")


def dft_good_size(length: int) -> int:
    return tc.smooth_good_size(length)


def make_dft_precomp(length: int, inverse: bool = False) -> dict[str, jax.Array]:
    n = int(length)
    if n <= 0:
        raise ValueError("make_dft_precomp requires length >= 1")
    sign = 1.0 if inverse else -1.0
    k = jnp.arange(n, dtype=tc.TRANSFORM_REAL_DTYPE)
    chirp = jnp.exp(1j * sign * el.PI * (k * k) / float(n))
    fft_size = dft_good_size(2 * n - 1)
    kernel = jnp.zeros((fft_size,), dtype=tc.TRANSFORM_COMPLEX_DTYPE)
    kernel_seq = jnp.exp(-1j * sign * el.PI * (k * k) / float(n))
    kernel = kernel.at[:n].set(kernel_seq)
    if n > 1:
        kernel = kernel.at[fft_size - (n - 1):].set(kernel_seq[1:][::-1])
    return {"chirp": chirp, "kernel_fft": jnp.fft.fft(kernel)}


def _resolve_dft_precomp(length: int, inverse: bool, precomp: dict[str, jax.Array] | None) -> dict[str, jax.Array]:
    if precomp is not None:
        return precomp
    return make_dft_precomp(length, inverse=inverse)


@partial(jax.jit, static_argnames=("inverse",))
def _dft_bluestein_apply(x: jax.Array, chirp: jax.Array, kernel_fft: jax.Array, inverse: bool = False) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    fft_size = kernel_fft.shape[0]
    a = jnp.zeros((fft_size,), dtype=tc.TRANSFORM_COMPLEX_DTYPE).at[:n].set(x * chirp)
    c = jnp.fft.ifft(jnp.fft.fft(a) * kernel_fft)
    y = chirp * c[:n]
    if inverse:
        y = y / float(n)
    return y


def _dft_matrix(n: int, inverse: bool = False) -> jax.Array:
    k = jnp.arange(n, dtype=tc.TRANSFORM_REAL_DTYPE)[:, None]
    t = jnp.arange(n, dtype=tc.TRANSFORM_REAL_DTYPE)[None, :]
    sign = 1.0 if inverse else -1.0
    expo = sign * 2.0j * el.PI * (k * t) / float(n)
    return jnp.exp(expo)


def as_acb_box(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float64)
    checks.check_last_dim(arr, 4, "dft.as_acb_box")
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
    ang = sign * el.TWO_PI * (k * t) / float(n)
    wr = jnp.cos(ang)
    wi = jnp.sin(ang)
    return jnp.stack([wr, wr, wi, wi], axis=-1)


def _acb_twiddle_matrix_interval(n: int, inverse: bool = False) -> jax.Array:
    k = jnp.arange(n, dtype=jnp.float64)[:, None]
    t = jnp.arange(n, dtype=jnp.float64)[None, :]
    sign = 1.0 if inverse else -1.0
    ang = sign * el.TWO_PI * (k * t) / float(n)
    wr = jnp.cos(ang)
    wi = jnp.sin(ang)
    wr_iv = di.interval(di._below(wr), di._above(wr))
    wi_iv = di.interval(di._below(wi), di._above(wi))
    return acb_box(wr_iv, wi_iv)


def _acb_half_width(iv: jax.Array) -> jax.Array:
    arr = di.as_interval(iv)
    return 0.5 * jnp.maximum(di.upper(arr) - di.lower(arr), 0.0)


def _acb_from_mid_and_half_width(mid: jax.Array, re_half_width: jax.Array, im_half_width: jax.Array) -> jax.Array:
    re_mid = jnp.real(mid)
    im_mid = jnp.imag(mid)
    return acb_box(
        di.interval(re_mid - re_half_width, re_mid + re_half_width),
        di.interval(im_mid - im_half_width, im_mid + im_half_width),
    )


def _twiddle_abs_parts(n: int) -> tuple[jax.Array, jax.Array]:
    k = jnp.arange(n, dtype=jnp.float64)[:, None]
    t = jnp.arange(n, dtype=jnp.float64)[None, :]
    ang = el.TWO_PI * (k * t) / float(n)
    return jnp.abs(jnp.cos(ang)), jnp.abs(jnp.sin(ang))


def _acb_pack_fft_basic(xb: jax.Array, midpoint_out: jax.Array, inverse: bool = False) -> jax.Array:
    re_half_width = _acb_half_width(acb_real(xb))
    im_half_width = _acb_half_width(acb_imag(xb))
    abs_cos, abs_sin = _twiddle_abs_parts(xb.shape[0])
    out_re_half_width = abs_cos @ re_half_width + abs_sin @ im_half_width
    out_im_half_width = abs_sin @ re_half_width + abs_cos @ im_half_width
    if inverse:
        scale = float(xb.shape[0])
        out_re_half_width = out_re_half_width / scale
        out_im_half_width = out_im_half_width / scale
    return _acb_from_mid_and_half_width(midpoint_out, out_re_half_width, out_im_half_width)


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


@partial(jax.jit, static_argnames=("inverse",))
def dft_bluestein_precomp(x: jax.Array, precomp: dict[str, jax.Array] | None = None, inverse: bool = False) -> jax.Array:
    x = _as_complex_vector(x)
    plan = _resolve_dft_precomp(x.shape[0], inverse=inverse, precomp=precomp)
    return _dft_bluestein_apply(x, plan["chirp"], plan["kernel_fft"], inverse=inverse)


@jax.jit
def dft_bluestein(x: jax.Array) -> jax.Array:
    return dft_bluestein_precomp(x, precomp=None, inverse=False)


@jax.jit
def idft_bluestein(x: jax.Array) -> jax.Array:
    return dft_bluestein_precomp(x, precomp=None, inverse=True)


@jax.jit
def dft_rad2(x: jax.Array) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    if _is_power_of_two(n):
        return jnp.fft.fft(x)
    return dft_bluestein(x)


@jax.jit
def idft_rad2(x: jax.Array) -> jax.Array:
    x = _as_complex_vector(x)
    n = x.shape[0]
    if _is_power_of_two(n):
        return jnp.fft.ifft(x)
    return idft_bluestein(x)


@jax.jit
def dft(x: jax.Array) -> jax.Array:
    return dft_rad2(x)


@jax.jit
def idft(x: jax.Array) -> jax.Array:
    return idft_rad2(x)


def dft_matvec_point(x: jax.Array, inverse: bool = False) -> jax.Array:
    return idft(x) if inverse else dft(x)


def dft_matvec_cached_prepare_point(length: int, inverse: bool = False) -> tc.DftMatvecPlan:
    n = int(length)
    if n <= 0:
        raise ValueError("dft_matvec_cached_prepare_point requires length >= 1")
    precomp = make_dft_precomp(n, inverse=inverse)
    return tc.DftMatvecPlan(chirp=precomp["chirp"], kernel_fft=precomp["kernel_fft"], length=n, inverse=bool(inverse))


@jax.jit
def dft_matvec_cached_apply_point(plan: tc.DftMatvecPlan, x: jax.Array) -> jax.Array:
    vec = tc.as_complex_vector(x, "dft.dft_matvec_cached_apply_point")
    checks.check_equal(plan.length, vec.shape[0], "dft.dft_matvec_cached_apply_point.length")
    if plan.inverse:
        return jnp.fft.ifft(vec) if _is_power_of_two(plan.length) else _dft_bluestein_apply(vec, plan.chirp, plan.kernel_fft, inverse=True)
    return jnp.fft.fft(vec) if _is_power_of_two(plan.length) else _dft_bluestein_apply(vec, plan.chirp, plan.kernel_fft, inverse=False)


def dft_matvec_cached_apply_point_with_diagnostics(
    plan: tc.DftMatvecPlan,
    x: jax.Array,
) -> tuple[jax.Array, dict[str, int | bool | str]]:
    out = dft_matvec_cached_apply_point(plan, x)
    return out, {
        "length": int(plan.length),
        "inverse": bool(plan.inverse),
        "method": "fft" if _is_power_of_two(plan.length) else "bluestein",
        "mode": "point",
    }


def dft_matvec_batch_fixed_point(x: jax.Array, inverse: bool = False) -> jax.Array:
    arr = tc.as_complex_array(x, "dft.dft_matvec_batch_fixed_point")
    checks.check_ndim(arr, 2, "dft.dft_matvec_batch_fixed_point")
    return jax.vmap(lambda row: dft_matvec_point(row, inverse=inverse))(arr)


def dft_matvec_batch_padded_point(x: jax.Array, *, pad_to: int, inverse: bool = False) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last(
        (tc.as_complex_array(x, "dft.dft_matvec_batch_padded_point"),),
        pad_to=pad_to,
    )
    out = dft_matvec_batch_fixed_point(*call_args, inverse=inverse)
    return kh.trim_batch_out(out, trim_n)


def dft_matvec_cached_apply_batch_fixed_point(plan: tc.DftMatvecPlan, x: jax.Array) -> jax.Array:
    arr = tc.as_complex_array(x, "dft.dft_matvec_cached_apply_batch_fixed_point")
    checks.check_ndim(arr, 2, "dft.dft_matvec_cached_apply_batch_fixed_point")
    return jax.vmap(lambda row: dft_matvec_cached_apply_point(plan, row))(arr)


def dft_matvec_cached_apply_batch_padded_point(plan: tc.DftMatvecPlan, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last(
        (tc.as_complex_array(x, "dft.dft_matvec_cached_apply_batch_padded_point"),),
        pad_to=pad_to,
    )
    out = dft_matvec_cached_apply_batch_fixed_point(plan, *call_args)
    return kh.trim_batch_out(out, trim_n)


def _apply_complex_axis_transform(x: jax.Array, axis: int, transform) -> jax.Array:
    moved = jnp.moveaxis(_as_complex_array(x), axis, -1)
    shape = moved.shape
    flat = moved.reshape((-1, shape[-1]))
    out = jax.vmap(transform)(flat)
    return jnp.moveaxis(out.reshape(shape), -1, axis)


@partial(jax.jit, static_argnames=("axes", "inverse"))
def dft_nd(x: jax.Array, axes: tuple[int, ...] | None = None, inverse: bool = False) -> jax.Array:
    arr = _as_complex_array(x)
    checks.check(arr.ndim >= 1, "dft.dft_nd")
    out = arr
    transform = idft if inverse else dft
    for axis in _canonical_axes(arr.ndim, axes):
        out = _apply_complex_axis_transform(out, axis, transform)
    return out


@partial(jax.jit, static_argnames=("axes",))
def idft_nd(x: jax.Array, axes: tuple[int, ...] | None = None) -> jax.Array:
    return dft_nd(x, axes=axes, inverse=True)


@jax.jit
def dft2(x: jax.Array) -> jax.Array:
    arr = _as_complex_array(x)
    checks.check_ndim(arr, 2, "dft.dft2")
    return dft_nd(arr, axes=(0, 1), inverse=False)


@jax.jit
def idft2(x: jax.Array) -> jax.Array:
    arr = _as_complex_array(x)
    checks.check_ndim(arr, 2, "dft.idft2")
    return dft_nd(arr, axes=(0, 1), inverse=True)


@jax.jit
def dft3(x: jax.Array) -> jax.Array:
    arr = _as_complex_array(x)
    checks.check_ndim(arr, 3, "dft.dft3")
    return dft_nd(arr, axes=(0, 1, 2), inverse=False)


@jax.jit
def idft3(x: jax.Array) -> jax.Array:
    arr = _as_complex_array(x)
    checks.check_ndim(arr, 3, "dft.idft3")
    return dft_nd(arr, axes=(0, 1, 2), inverse=True)


@partial(jax.jit, static_argnames=("cyc",))
def dft_prod(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    x = _as_complex_vector(x)
    n = 1
    for m in cyc:
        n *= int(m)
    checks.check_equal(n, x.shape[0], "dft.dft_prod")
    return dft(x)


@jax.jit
def convol_circular_naive(f: jax.Array, g: jax.Array) -> jax.Array:
    f = _as_complex_vector(f)
    g = _as_complex_vector(g)
    checks.check_equal(f.shape[0], g.shape[0], "dft.convol_circular_naive")
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
    checks.check_equal(f.shape[0], g.shape[0], "dft.convol_circular_dft")
    return idft(dft(f) * dft(g))


@jax.jit
def convol_circular_rad2(f: jax.Array, g: jax.Array) -> jax.Array:
    f = _as_complex_vector(f)
    g = _as_complex_vector(g)
    checks.check_equal(f.shape[0], g.shape[0], "dft.convol_circular_rad2")
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


@partial(jax.jit, static_argnames=("inverse",))
def acb_dft_naive_rigorous(x: jax.Array, inverse: bool = False) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    w = _acb_twiddle_matrix_interval(n, inverse=inverse)

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
    midpoint = acb_core.acb_midpoint(xb)
    if _is_power_of_two(n):
        midpoint_out = jnp.fft.fft(midpoint)
    else:
        midpoint_out = dft_bluestein(midpoint)
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=False)


@jax.jit
def acb_idft_naive_rigorous(x: jax.Array) -> jax.Array:
    return acb_dft_naive_rigorous(x, inverse=True)


@jax.jit
def acb_dft_rad2_rigorous(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_naive_rigorous(xb, inverse=False)
    return acb_dft_naive_rigorous(xb, inverse=False)


@jax.jit
def acb_idft_rad2(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    midpoint = acb_core.acb_midpoint(xb)
    if _is_power_of_two(n):
        midpoint_out = jnp.fft.ifft(midpoint)
    else:
        midpoint_out = idft_bluestein(midpoint)
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=True)


@jax.jit
def acb_idft_rad2_rigorous(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_naive_rigorous(xb, inverse=True)
    return acb_dft_naive_rigorous(xb, inverse=True)


@jax.jit
def acb_dft(x: jax.Array) -> jax.Array:
    return acb_dft_rad2(x)


@jax.jit
def acb_idft(x: jax.Array) -> jax.Array:
    return acb_idft_rad2(x)


def dft_matvec_basic(x: jax.Array, inverse: bool = False) -> jax.Array:
    return acb_idft(x) if inverse else acb_dft(x)


def dft_matvec_cached_prepare_basic(length: int, inverse: bool = False) -> tc.DftMatvecPlan:
    return dft_matvec_cached_prepare_point(length, inverse=inverse)


@jax.jit
def dft_matvec_cached_apply_basic(plan: tc.DftMatvecPlan, x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    checks.check_equal(plan.length, xb.shape[0], "dft.dft_matvec_cached_apply_basic.length")
    midpoint = acb_core.acb_midpoint(xb)
    midpoint_out = dft_matvec_cached_apply_point(plan, midpoint)
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=plan.inverse)


def dft_matvec_cached_apply_basic_with_diagnostics(
    plan: tc.DftMatvecPlan,
    x: jax.Array,
) -> tuple[jax.Array, dict[str, int | bool | str]]:
    out = dft_matvec_cached_apply_basic(plan, x)
    return out, {
        "length": int(plan.length),
        "inverse": bool(plan.inverse),
        "method": "fft" if _is_power_of_two(plan.length) else "bluestein",
        "mode": "basic",
    }


def dft_matvec_batch_fixed_basic(x: jax.Array, inverse: bool = False) -> jax.Array:
    xb = as_acb_box(x)
    checks.check_ndim(xb, 3, "dft.dft_matvec_batch_fixed_basic")
    return jax.vmap(lambda row: dft_matvec_basic(row, inverse=inverse))(xb)


def dft_matvec_batch_padded_basic(x: jax.Array, *, pad_to: int, inverse: bool = False) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((as_acb_box(x),), pad_to=pad_to)
    out = dft_matvec_batch_fixed_basic(*call_args, inverse=inverse)
    return kh.trim_batch_out(out, trim_n)


def dft_matvec_cached_apply_batch_fixed_basic(plan: tc.DftMatvecPlan, x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    checks.check_ndim(xb, 3, "dft.dft_matvec_cached_apply_batch_fixed_basic")
    return jax.vmap(lambda row: dft_matvec_cached_apply_basic(plan, row))(xb)


def dft_matvec_cached_apply_batch_padded_basic(plan: tc.DftMatvecPlan, x: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, trim_n = kh.pad_mixed_batch_args_repeat_last((as_acb_box(x),), pad_to=pad_to)
    out = dft_matvec_cached_apply_batch_fixed_basic(plan, *call_args)
    return kh.trim_batch_out(out, trim_n)


def _apply_acb_axis_transform(x: jax.Array, axis: int, transform) -> jax.Array:
    box = as_acb_box(x)
    moved = jnp.moveaxis(box, axis, -2)
    shape = moved.shape
    flat = moved.reshape((-1, shape[-2], 4))
    out = jax.vmap(transform)(flat)
    return jnp.moveaxis(out.reshape(shape), -2, axis)


@partial(jax.jit, static_argnames=("axes", "rigorous", "inverse"))
def acb_dft_nd(
    x: jax.Array,
    axes: tuple[int, ...] | None = None,
    rigorous: bool = False,
    inverse: bool = False,
) -> jax.Array:
    box = as_acb_box(x)
    checks.check(box.ndim >= 2, "dft.acb_dft_nd")
    out = box
    if rigorous:
        transform = acb_idft_rigorous if inverse else acb_dft_rigorous
    else:
        transform = acb_idft if inverse else acb_dft
    for axis in _canonical_axes(box.ndim - 1, axes):
        out = _apply_acb_axis_transform(out, axis, transform)
    return out


@partial(jax.jit, static_argnames=("axes", "rigorous"))
def acb_idft_nd(x: jax.Array, axes: tuple[int, ...] | None = None, rigorous: bool = False) -> jax.Array:
    return acb_dft_nd(x, axes=axes, rigorous=rigorous, inverse=True)


@jax.jit
def acb_dft2(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    checks.check(box.ndim == 3, "dft.acb_dft2")
    return acb_dft_nd(box, axes=(0, 1), rigorous=False, inverse=False)


@jax.jit
def acb_idft2(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    checks.check(box.ndim == 3, "dft.acb_idft2")
    return acb_dft_nd(box, axes=(0, 1), rigorous=False, inverse=True)


@jax.jit
def acb_dft3(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    checks.check(box.ndim == 4, "dft.acb_dft3")
    return acb_dft_nd(box, axes=(0, 1, 2), rigorous=False, inverse=False)


@jax.jit
def acb_idft3(x: jax.Array) -> jax.Array:
    box = as_acb_box(x)
    checks.check(box.ndim == 4, "dft.acb_idft3")
    return acb_dft_nd(box, axes=(0, 1, 2), rigorous=False, inverse=True)


@jax.jit
def acb_dft_rigorous(x: jax.Array) -> jax.Array:
    return acb_dft_rad2_rigorous(x)


@jax.jit
def acb_idft_rigorous(x: jax.Array) -> jax.Array:
    return acb_idft_rad2_rigorous(x)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_prod(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    xb = as_acb_box(x)
    n = 1
    for m in cyc:
        n *= int(m)
    checks.check_equal(n, xb.shape[0], "dft.acb_dft_prod")
    return acb_dft(xb)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_prod_rigorous(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    xb = as_acb_box(x)
    n = 1
    for m in cyc:
        n *= int(m)
    checks.check_equal(n, xb.shape[0], "dft.acb_dft_prod_rigorous")
    return acb_dft_rigorous(xb)


@jax.jit
def acb_convol_circular_naive(f: jax.Array, g: jax.Array) -> jax.Array:
    fb = as_acb_box(f)
    gb = as_acb_box(g)
    checks.check_equal(fb.shape[0], gb.shape[0], "dft.acb_convol_circular_naive")
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
def acb_convol_circular_naive_rigorous(f: jax.Array, g: jax.Array) -> jax.Array:
    fb = as_acb_box(f)
    gb = as_acb_box(g)
    checks.check_equal(fb.shape[0], gb.shape[0], "dft.acb_convol_circular_naive_rigorous")
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
    checks.check_equal(fb.shape[0], gb.shape[0], "dft.acb_convol_circular_dft")
    return acb_idft(acb_mul_vec(acb_dft(fb), acb_dft(gb)))


@jax.jit
def acb_convol_circular_dft_rigorous(f: jax.Array, g: jax.Array) -> jax.Array:
    fb = as_acb_box(f)
    gb = as_acb_box(g)
    checks.check_equal(fb.shape[0], gb.shape[0], "dft.acb_convol_circular_dft_rigorous")
    return acb_idft_rigorous(acb_mul_vec_rigorous(acb_dft_rigorous(fb), acb_dft_rigorous(gb)))


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


def acb_mul_vec_rigorous(x: jax.Array, y: jax.Array) -> jax.Array:
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
def acb_convol_circular_rad2_rigorous(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_idft_rad2_rigorous(acb_mul_vec_rigorous(acb_dft_rad2_rigorous(f), acb_dft_rad2_rigorous(g)))


@jax.jit
def acb_convol_circular(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_rad2(f, g)


@jax.jit
def acb_convol_circular_rigorous(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_rad2_rigorous(f, g)


@jax.jit
def acb_dft_bluestein(x: jax.Array) -> jax.Array:
    xb = as_acb_box(x)
    midpoint_out = dft_bluestein(acb_core.acb_midpoint(xb))
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=False)


def acb_dft_bluestein_precomp(x: jax.Array, precomp: dict[str, jax.Array] | None = None) -> jax.Array:
    xb = as_acb_box(x)
    midpoint_out = dft_bluestein_precomp(acb_core.acb_midpoint(xb), precomp=precomp, inverse=False)
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=False)


@jax.jit
def acb_dft_convol(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular(f, g)


@jax.jit
def acb_dft_convol_dft(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_dft(f, g)


@jax.jit
def acb_dft_convol_mullow(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_naive(f, g)


@jax.jit
def acb_dft_convol_naive(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_naive(f, g)


@jax.jit
def acb_dft_convol_rad2(f: jax.Array, g: jax.Array) -> jax.Array:
    return acb_convol_circular_rad2(f, g)


@jax.jit
def acb_dft_convol_rad2_precomp(f: jax.Array, g: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_convol_rad2(f, g)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_crt(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    return acb_dft_prod(x, cyc)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_crt_precomp(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_crt(x, cyc)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_cyc(x: jax.Array, cyc: tuple[int, ...]) -> jax.Array:
    return acb_dft_prod(x, cyc)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_cyc_precomp(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_cyc(x, cyc)


@jax.jit
def acb_dft_inverse(x: jax.Array) -> jax.Array:
    return acb_idft(x)


@jax.jit
def acb_dft_inverse_precomp(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    xb = as_acb_box(x)
    if _is_power_of_two(xb.shape[0]):
        return acb_idft_rad2(xb)
    midpoint_out = dft_bluestein_precomp(acb_core.acb_midpoint(xb), precomp=precomp, inverse=True)
    return _acb_pack_fft_basic(xb, midpoint_out, inverse=True)


@jax.jit
def acb_dft_inverse_rad2_precomp(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_inverse_precomp(x, precomp)


@jax.jit
def acb_dft_inverse_rad2_precomp_inplace(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_inverse_rad2_precomp(x, precomp)


@partial(jax.jit, static_argnames=("inverse",))
def acb_dft_naive_precomp(x: jax.Array, inverse: bool = False, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_naive(x, inverse=inverse)


@jax.jit
def acb_dft_precomp(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_rad2(xb)
    return acb_dft_bluestein_precomp(xb, precomp)


@partial(jax.jit, static_argnames=("cyc",))
def acb_dft_prod_precomp(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_prod(x, cyc)


@jax.jit
def acb_dft_rad2_inplace(x: jax.Array) -> jax.Array:
    return acb_dft_rad2(x)


@jax.jit
def acb_dft_rad2_inplace_threaded(x: jax.Array) -> jax.Array:
    return acb_dft_rad2(x)


@jax.jit
def acb_dft_rad2_precomp(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    xb = as_acb_box(x)
    n = xb.shape[0]
    if _is_power_of_two(n):
        return acb_dft_rad2(xb)
    return acb_dft_bluestein_precomp(xb, precomp)


@jax.jit
def acb_dft_rad2_precomp_inplace(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_rad2_precomp(x, precomp)


@jax.jit
def acb_dft_rad2_precomp_inplace_threaded(x: jax.Array, precomp: jax.Array | None = None) -> jax.Array:
    return acb_dft_rad2_precomp(x, precomp)


@jax.jit
def acb_dft_step(x: jax.Array) -> jax.Array:
    return acb_dft(x)


dft_naive_jit = jax.jit(dft_naive, static_argnames=("inverse",))
idft_naive_jit = jax.jit(idft_naive)
dft_bluestein_jit = jax.jit(dft_bluestein)
idft_bluestein_jit = jax.jit(idft_bluestein)
dft_rad2_jit = jax.jit(dft_rad2)
idft_rad2_jit = jax.jit(idft_rad2)
dft_jit = jax.jit(dft)
idft_jit = jax.jit(idft)
dft_matvec_point_jit = jax.jit(dft_matvec_point, static_argnames=("inverse",))
dft_matvec_cached_apply_point_jit = jax.jit(dft_matvec_cached_apply_point)
dft_matvec_batch_fixed_point_jit = jax.jit(dft_matvec_batch_fixed_point, static_argnames=("inverse",))
dft_matvec_cached_apply_batch_fixed_point_jit = jax.jit(dft_matvec_cached_apply_batch_fixed_point)
dft_nd_jit = jax.jit(dft_nd, static_argnames=("axes", "inverse"))
idft_nd_jit = jax.jit(idft_nd, static_argnames=("axes",))
dft2_jit = jax.jit(dft2)
idft2_jit = jax.jit(idft2)
dft3_jit = jax.jit(dft3)
idft3_jit = jax.jit(idft3)
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
dft_matvec_basic_jit = jax.jit(dft_matvec_basic, static_argnames=("inverse",))
dft_matvec_cached_apply_basic_jit = jax.jit(dft_matvec_cached_apply_basic)
dft_matvec_batch_fixed_basic_jit = jax.jit(dft_matvec_batch_fixed_basic, static_argnames=("inverse",))
dft_matvec_cached_apply_batch_fixed_basic_jit = jax.jit(dft_matvec_cached_apply_batch_fixed_basic)
acb_dft_nd_jit = jax.jit(acb_dft_nd, static_argnames=("axes", "rigorous", "inverse"))
acb_idft_nd_jit = jax.jit(acb_idft_nd, static_argnames=("axes", "rigorous"))
acb_dft2_jit = jax.jit(acb_dft2)
acb_idft2_jit = jax.jit(acb_idft2)
acb_dft3_jit = jax.jit(acb_dft3)
acb_idft3_jit = jax.jit(acb_idft3)
acb_dft_prod_jit = jax.jit(acb_dft_prod, static_argnames=("cyc",))
acb_convol_circular_naive_jit = jax.jit(acb_convol_circular_naive)
acb_convol_circular_dft_jit = jax.jit(acb_convol_circular_dft)
acb_convol_circular_rad2_jit = jax.jit(acb_convol_circular_rad2)
acb_convol_circular_jit = jax.jit(acb_convol_circular)

@partial(jax.jit, static_argnames=("inverse", "prec_bits"))
def acb_dft_naive_prec(x: jax.Array, inverse: bool = False, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_naive(x, inverse=inverse), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_idft_naive_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_idft_naive(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_idft_rad2_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_idft_rad2(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_idft_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_idft(x), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_prod_prec(x: jax.Array, cyc: tuple[int, ...], prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_prod(x, cyc), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_convol_circular_naive_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_convol_circular_naive(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_convol_circular_dft_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_convol_circular_dft(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_convol_circular_rad2_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_convol_circular_rad2(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_convol_circular_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_convol_circular(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_mul_vec_prec(x: jax.Array, y: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_mul_vec(x, y), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_bluestein_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_bluestein(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_bluestein_precomp_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_bluestein_precomp(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_dft_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol_dft(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_mullow_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol_mullow(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_naive_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol_naive(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_rad2_prec(f: jax.Array, g: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol_rad2(f, g), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_convol_rad2_precomp_prec(f: jax.Array, g: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_convol_rad2_precomp(f, g, precomp), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_crt_prec(x: jax.Array, cyc: tuple[int, ...], prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_crt(x, cyc), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_crt_precomp_prec(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_crt_precomp(x, cyc, precomp), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_cyc_prec(x: jax.Array, cyc: tuple[int, ...], prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_cyc(x, cyc), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_cyc_precomp_prec(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_cyc_precomp(x, cyc, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_inverse_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_inverse(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_inverse_precomp_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_inverse_precomp(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_inverse_rad2_precomp_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_inverse_rad2_precomp(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_inverse_rad2_precomp_inplace_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_inverse_rad2_precomp_inplace(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("inverse", "prec_bits"))
def acb_dft_naive_precomp_prec(x: jax.Array, inverse: bool = False, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_naive_precomp(x, inverse=inverse, precomp=precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_precomp_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_precomp(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("cyc", "prec_bits"))
def acb_dft_prod_precomp_prec(x: jax.Array, cyc: tuple[int, ...], precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_prod_precomp(x, cyc, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_inplace_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2_inplace(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_inplace_threaded_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2_inplace_threaded(x), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_precomp_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2_precomp(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_precomp_inplace_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2_precomp_inplace(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_rad2_precomp_inplace_threaded_prec(x: jax.Array, precomp: jax.Array | None = None, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_rad2_precomp_inplace_threaded(x, precomp), prec_bits)


@partial(jax.jit, static_argnames=("prec_bits",))
def acb_dft_step_prec(x: jax.Array, prec_bits: int = di.DEFAULT_PREC_BITS) -> jax.Array:
    return acb_core.acb_box_round_prec(acb_dft_step(x), prec_bits)


__all__ = [
    "dft_good_size",
    "make_dft_precomp",
    "dft_naive",
    "idft_naive",
    "dft_bluestein_precomp",
    "dft_bluestein",
    "idft_bluestein",
    "dft_rad2",
    "idft_rad2",
    "dft",
    "idft",
    "dft_matvec_point",
    "dft_matvec_cached_prepare_point",
    "dft_matvec_cached_apply_point",
    "dft_matvec_cached_apply_point_with_diagnostics",
    "dft_matvec_batch_fixed_point",
    "dft_matvec_batch_padded_point",
    "dft_matvec_cached_apply_batch_fixed_point",
    "dft_matvec_cached_apply_batch_padded_point",
    "dft_nd",
    "idft_nd",
    "dft2",
    "idft2",
    "dft3",
    "idft3",
    "dft_prod",
    "convol_circular_naive",
    "convol_circular_dft",
    "convol_circular_rad2",
    "convol_circular",
    "dft_naive_jit",
    "idft_naive_jit",
    "dft_bluestein_jit",
    "idft_bluestein_jit",
    "dft_rad2_jit",
    "idft_rad2_jit",
    "dft_jit",
    "idft_jit",
    "dft_matvec_point_jit",
    "dft_matvec_cached_apply_point_jit",
    "dft_matvec_batch_fixed_point_jit",
    "dft_matvec_cached_apply_batch_fixed_point_jit",
    "dft_nd_jit",
    "idft_nd_jit",
    "dft2_jit",
    "idft2_jit",
    "dft3_jit",
    "idft3_jit",
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
    "acb_dft_naive_rigorous",
    "acb_idft_naive",
    "acb_dft_rad2",
    "acb_dft_rad2_rigorous",
    "acb_idft_rad2",
    "acb_dft",
    "acb_dft_rigorous",
    "acb_idft",
    "acb_idft_rigorous",
    "dft_matvec_basic",
    "dft_matvec_cached_prepare_basic",
    "dft_matvec_cached_apply_basic",
    "dft_matvec_cached_apply_basic_with_diagnostics",
    "dft_matvec_batch_fixed_basic",
    "dft_matvec_batch_padded_basic",
    "dft_matvec_cached_apply_batch_fixed_basic",
    "dft_matvec_cached_apply_batch_padded_basic",
    "acb_dft_nd",
    "acb_idft_nd",
    "acb_dft2",
    "acb_idft2",
    "acb_dft3",
    "acb_idft3",
    "acb_dft_prod",
    "acb_dft_prod_rigorous",
    "acb_convol_circular_naive",
    "acb_convol_circular_naive_rigorous",
    "acb_convol_circular_dft",
    "acb_convol_circular_dft_rigorous",
    "acb_convol_circular_rad2",
    "acb_convol_circular_rad2_rigorous",
    "acb_convol_circular",
    "acb_convol_circular_rigorous",
    "acb_idft_naive_rigorous",
    "acb_idft_rad2_rigorous",
    "acb_mul_vec_rigorous",
    "acb_dft_naive_jit",
    "acb_idft_naive_jit",
    "acb_dft_rad2_jit",
    "acb_idft_rad2_jit",
    "acb_dft_jit",
    "acb_idft_jit",
    "dft_matvec_basic_jit",
    "dft_matvec_cached_apply_basic_jit",
    "dft_matvec_batch_fixed_basic_jit",
    "dft_matvec_cached_apply_batch_fixed_basic_jit",
    "acb_dft_nd_jit",
    "acb_idft_nd_jit",
    "acb_dft2_jit",
    "acb_idft2_jit",
    "acb_dft3_jit",
    "acb_idft3_jit",
    "acb_dft_prod_jit",
    "acb_convol_circular_naive_jit",
    "acb_convol_circular_dft_jit",
    "acb_convol_circular_rad2_jit",
    "acb_convol_circular_jit",
    "acb_dft_naive_prec",
    "acb_idft_naive_prec",
    "acb_dft_rad2_prec",
    "acb_idft_rad2_prec",
    "acb_dft_prec",
    "acb_idft_prec",
    "acb_dft_prod_prec",
    "acb_convol_circular_naive_prec",
    "acb_convol_circular_dft_prec",
    "acb_convol_circular_rad2_prec",
    "acb_convol_circular_prec",
    "acb_mul_vec_prec",
    "acb_dft_bluestein",
    "acb_dft_bluestein_precomp",
    "acb_dft_convol",
    "acb_dft_convol_dft",
    "acb_dft_convol_mullow",
    "acb_dft_convol_naive",
    "acb_dft_convol_rad2",
    "acb_dft_convol_rad2_precomp",
    "acb_dft_crt",
    "acb_dft_crt_precomp",
    "acb_dft_cyc",
    "acb_dft_cyc_precomp",
    "acb_dft_inverse",
    "acb_dft_inverse_precomp",
    "acb_dft_inverse_rad2_precomp",
    "acb_dft_inverse_rad2_precomp_inplace",
    "acb_dft_naive_precomp",
    "acb_dft_precomp",
    "acb_dft_prod_precomp",
    "acb_dft_rad2_inplace",
    "acb_dft_rad2_inplace_threaded",
    "acb_dft_rad2_precomp",
    "acb_dft_rad2_precomp_inplace",
    "acb_dft_rad2_precomp_inplace_threaded",
    "acb_dft_step",
    "acb_dft_bluestein_prec",
    "acb_dft_bluestein_precomp_prec",
    "acb_dft_convol_prec",
    "acb_dft_convol_dft_prec",
    "acb_dft_convol_mullow_prec",
    "acb_dft_convol_naive_prec",
    "acb_dft_convol_rad2_prec",
    "acb_dft_convol_rad2_precomp_prec",
    "acb_dft_crt_prec",
    "acb_dft_crt_precomp_prec",
    "acb_dft_cyc_prec",
    "acb_dft_cyc_precomp_prec",
    "acb_dft_inverse_prec",
    "acb_dft_inverse_precomp_prec",
    "acb_dft_inverse_rad2_precomp_prec",
    "acb_dft_inverse_rad2_precomp_inplace_prec",
    "acb_dft_naive_precomp_prec",
    "acb_dft_precomp_prec",
    "acb_dft_prod_precomp_prec",
    "acb_dft_rad2_inplace_prec",
    "acb_dft_rad2_inplace_threaded_prec",
    "acb_dft_rad2_precomp_prec",
    "acb_dft_rad2_precomp_inplace_prec",
    "acb_dft_rad2_precomp_inplace_threaded_prec",
    "acb_dft_step_prec",
]
