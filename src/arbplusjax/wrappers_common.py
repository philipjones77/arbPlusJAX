from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di
from . import precision
from . import checks

jax.config.update("jax_enable_x64", True)


def resolve_prec_bits(dps: int | None, prec_bits: int | None) -> int:
    if prec_bits is not None:
        return int(prec_bits)
    if dps is not None:
        return precision.dps_to_bits(int(dps))
    return precision.get_prec_bits()


def inflate_interval(x: jax.Array, prec_bits: int, adaptive: bool) -> jax.Array:
    x = di.as_interval(x)
    lo = di.lower(x)
    hi = di.upper(x)
    mid = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    scale = 1.0 + jnp.abs(mid) + (2.0 if adaptive else 1.0) * rad
    extra = eps * scale
    return di.interval(lo - extra, hi + extra)


def inflate_acb(x: jax.Array, prec_bits: int, adaptive: bool) -> jax.Array:
    r = inflate_interval(acb_core.acb_real(x), prec_bits, adaptive)
    i = inflate_interval(acb_core.acb_imag(x), prec_bits, adaptive)
    return acb_core.acb_box(r, i)


def dispatch_mode(
    impl: str,
    base_fn: Callable[..., jax.Array],
    rig_fn: Callable[..., jax.Array] | None,
    adapt_fn: Callable[..., jax.Array] | None,
    is_acb: bool,
    prec_bits: int,
    args: tuple,
    kwargs: dict,
) -> jax.Array:
    checks.check_in_set(impl, ("baseline", "rigorous", "adaptive"), "wrappers_common.impl")
    if impl == "baseline":
        return base_fn(*args, prec_bits=prec_bits, **kwargs)
    if impl == "rigorous":
        if rig_fn is not None:
            return rig_fn(*args, prec_bits=prec_bits, **kwargs)
        out = base_fn(*args, prec_bits=prec_bits, **kwargs)
        return inflate_acb(out, prec_bits, adaptive=False) if is_acb else inflate_interval(out, prec_bits, adaptive=False)
    if impl == "adaptive":
        if adapt_fn is not None:
            return adapt_fn(*args, prec_bits=prec_bits, **kwargs)
        out = base_fn(*args, prec_bits=prec_bits, **kwargs)
        return inflate_acb(out, prec_bits, adaptive=True) if is_acb else inflate_interval(out, prec_bits, adaptive=True)
    return base_fn(*args, prec_bits=prec_bits, **kwargs)


def _flatten_arrays(arrs: list[jax.Array]) -> tuple[jax.Array, list[tuple[int, ...]], list[int]]:
    flats = [jnp.ravel(a) for a in arrs]
    sizes = [int(a.size) for a in flats]
    shapes = [a.shape for a in arrs]
    if not flats:
        return jnp.zeros((0,), dtype=jnp.float64), shapes, sizes
    return jnp.concatenate(flats, axis=0), shapes, sizes


def _unflatten(vec: jax.Array, shapes: list[tuple[int, ...]], sizes: list[int]) -> list[jax.Array]:
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        part = vec[offset:offset + size]
        out.append(jnp.reshape(part, shape))
        offset += size
    return out


def _interval_mid_rad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return di.midpoint(x), di.ubound_radius(x)


def _box_mid_rad(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    re_mid, re_rad = _interval_mid_rad(re)
    im_mid, im_rad = _interval_mid_rad(im)
    return re_mid, im_mid, re_rad, im_rad


def _interval_from_mid(mid: jax.Array) -> jax.Array:
    return di.interval(mid, mid)


def _box_from_mid(mid: jax.Array) -> jax.Array:
    re = jnp.real(mid)
    im = jnp.imag(mid)
    return acb_core.acb_box(di.interval(re, re), di.interval(im, im))


def rigorous_interval_kernel(
    fn,
    interval_args: tuple[jax.Array, ...],
    prec_bits: int,
    **kwargs,
) -> jax.Array:
    mids = []
    rads = []
    for arg in interval_args:
        mid, rad = _interval_mid_rad(arg)
        mids.append(mid)
        rads.append(rad)
    mid_vec, shapes, sizes = _flatten_arrays(mids)
    rad_vec, _, _ = _flatten_arrays(rads)

    def f_flat(vec):
        parts = _unflatten(vec, shapes, sizes)
        call_args = [_interval_from_mid(p) for p in parts]
        out = fn(*call_args, **kwargs)
        return di.midpoint(out)

    y0 = f_flat(mid_vec)
    y0_flat, y_shapes, y_sizes = _flatten_arrays([jnp.asarray(y0, dtype=jnp.float64)])
    if y0_flat.size == 0:
        return di.interval(y0, y0)

    jac = jax.jacfwd(f_flat)(mid_vec)
    jac_flat = jnp.reshape(jac, (y0_flat.size, mid_vec.size))
    bound = jnp.sum(jnp.abs(jac_flat) * rad_vec[None, :], axis=-1)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    bound = bound + eps
    lo = y0_flat - bound
    hi = y0_flat + bound
    out_flat = di.interval(di._below(lo), di._above(hi))
    out = jnp.reshape(out_flat, y0.shape + (2,)) if y0.shape != () else out_flat
    return out


def adaptive_interval_kernel(
    fn,
    interval_args: tuple[jax.Array, ...],
    prec_bits: int,
    **kwargs,
) -> jax.Array:
    mids = []
    rads = []
    for arg in interval_args:
        mid, rad = _interval_mid_rad(arg)
        mids.append(mid)
        rads.append(rad)
    mid_vec, shapes, sizes = _flatten_arrays(mids)
    rad_vec, _, _ = _flatten_arrays(rads)

    def f_flat(vec):
        parts = _unflatten(vec, shapes, sizes)
        call_args = [_interval_from_mid(p) for p in parts]
        out = fn(*call_args, **kwargs)
        return di.midpoint(out)

    y0 = f_flat(mid_vec)
    y0_flat = jnp.ravel(jnp.asarray(y0, dtype=jnp.float64))
    if y0_flat.size == 0:
        return di.interval(y0, y0)

    samples = jnp.stack([mid_vec, mid_vec + rad_vec, mid_vec - rad_vec], axis=0)
    ys = jax.vmap(f_flat)(samples)
    ys_flat = jnp.reshape(ys, (samples.shape[0], y0_flat.size))
    dev = jnp.max(jnp.abs(ys_flat - y0_flat[None, :]), axis=0)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    bound = dev + eps
    lo = y0_flat - bound
    hi = y0_flat + bound
    out_flat = di.interval(di._below(lo), di._above(hi))
    out = jnp.reshape(out_flat, y0.shape + (2,)) if y0.shape != () else out_flat
    return out


def rigorous_acb_kernel(
    fn,
    box_args: tuple[jax.Array, ...],
    prec_bits: int,
    **kwargs,
) -> jax.Array:
    re_mids = []
    im_mids = []
    re_rads = []
    im_rads = []
    for arg in box_args:
        re_mid, im_mid, re_rad, im_rad = _box_mid_rad(arg)
        re_mids.append(re_mid)
        im_mids.append(im_mid)
        re_rads.append(re_rad)
        im_rads.append(im_rad)
    mid_vec, shapes, sizes = _flatten_arrays(re_mids + im_mids)
    rad_vec, _, _ = _flatten_arrays(re_rads + im_rads)

    n = len(re_mids)

    def f_flat(vec):
        parts = _unflatten(vec, shapes, sizes)
        re_parts = parts[:n]
        im_parts = parts[n:]
        mids = [re + 1j * im for re, im in zip(re_parts, im_parts)]
        call_args = [_box_from_mid(m) for m in mids]
        out = fn(*call_args, **kwargs)
        return acb_core.acb_midpoint(out)

    z0 = f_flat(mid_vec)
    z0_re = jnp.ravel(jnp.real(z0))
    z0_im = jnp.ravel(jnp.imag(z0))
    if z0_re.size == 0:
        return acb_core.acb_box(di.interval(z0_re, z0_re), di.interval(z0_im, z0_im))

    def f_stack(vec):
        z = f_flat(vec)
        return jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)

    jac = jax.jacfwd(f_stack)(mid_vec)
    jac = jnp.reshape(jac, (z0_re.size, 2, mid_vec.size))
    bound_re = jnp.sum(jnp.abs(jac[:, 0, :]) * rad_vec[None, :], axis=-1)
    bound_im = jnp.sum(jnp.abs(jac[:, 1, :]) * rad_vec[None, :], axis=-1)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    bound_re = bound_re + eps
    bound_im = bound_im + eps

    lo_re = z0_re - bound_re
    hi_re = z0_re + bound_re
    lo_im = z0_im - bound_im
    hi_im = z0_im + bound_im
    re_iv = di.interval(di._below(lo_re), di._above(hi_re))
    im_iv = di.interval(di._below(lo_im), di._above(hi_im))
    re_iv = jnp.reshape(re_iv, z0.shape + (2,))
    im_iv = jnp.reshape(im_iv, z0.shape + (2,))
    return acb_core.acb_box(re_iv, im_iv)


def adaptive_acb_kernel(
    fn,
    box_args: tuple[jax.Array, ...],
    prec_bits: int,
    **kwargs,
) -> jax.Array:
    re_mids = []
    im_mids = []
    re_rads = []
    im_rads = []
    for arg in box_args:
        re_mid, im_mid, re_rad, im_rad = _box_mid_rad(arg)
        re_mids.append(re_mid)
        im_mids.append(im_mid)
        re_rads.append(re_rad)
        im_rads.append(im_rad)
    mid_vec, shapes, sizes = _flatten_arrays(re_mids + im_mids)
    rad_vec, _, _ = _flatten_arrays(re_rads + im_rads)

    n = len(re_mids)

    def f_flat(vec):
        parts = _unflatten(vec, shapes, sizes)
        re_parts = parts[:n]
        im_parts = parts[n:]
        mids = [re + 1j * im for re, im in zip(re_parts, im_parts)]
        call_args = [_box_from_mid(m) for m in mids]
        out = fn(*call_args, **kwargs)
        return acb_core.acb_midpoint(out)

    z0 = f_flat(mid_vec)
    z0_re = jnp.ravel(jnp.real(z0))
    z0_im = jnp.ravel(jnp.imag(z0))
    if z0_re.size == 0:
        return acb_core.acb_box(di.interval(z0_re, z0_re), di.interval(z0_im, z0_im))

    samples = jnp.stack([mid_vec, mid_vec + rad_vec, mid_vec - rad_vec], axis=0)
    zs = jax.vmap(f_flat)(samples)
    zs_re = jnp.reshape(jnp.real(zs), (samples.shape[0], z0_re.size))
    zs_im = jnp.reshape(jnp.imag(zs), (samples.shape[0], z0_im.size))
    dev_re = jnp.max(jnp.abs(zs_re - z0_re[None, :]), axis=0)
    dev_im = jnp.max(jnp.abs(zs_im - z0_im[None, :]), axis=0)
    eps = jnp.exp2(-jnp.float64(prec_bits))
    bound_re = dev_re + eps
    bound_im = dev_im + eps

    lo_re = z0_re - bound_re
    hi_re = z0_re + bound_re
    lo_im = z0_im - bound_im
    hi_im = z0_im + bound_im
    re_iv = di.interval(di._below(lo_re), di._above(hi_re))
    im_iv = di.interval(di._below(lo_im), di._above(hi_im))
    re_iv = jnp.reshape(re_iv, z0.shape + (2,))
    im_iv = jnp.reshape(im_iv, z0.shape + (2,))
    return acb_core.acb_box(re_iv, im_iv)
