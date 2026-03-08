from __future__ import annotations

import jax
import jax.numpy as jnp

from . import acb_core
from . import checks
from . import double_interval as di

jax.config.update("jax_enable_x64", True)


def as_interval_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check_tail_shape(arr, (2, 2, 2), label)
    return arr


def as_interval_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], f"{label}.square")
    return arr


def as_interval_vector(x: jax.Array, label: str) -> jax.Array:
    arr = di.as_interval(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 2, f"{label}.tail")
    return arr


def as_box_mat_2x2(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check_tail_shape(arr, (2, 2, 4), label)
    return arr


def as_box_matrix(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 3, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    checks.check_equal(arr.shape[-2], arr.shape[-3], f"{label}.square")
    return arr


def as_box_vector(x: jax.Array, label: str) -> jax.Array:
    arr = acb_core.as_acb_box(x)
    checks.check(arr.ndim >= 2, f"{label}.ndim")
    checks.check_equal(arr.shape[-1], 4, f"{label}.tail")
    return arr


def full_interval_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    return di.interval(-jnp.inf * t, jnp.inf * t)


def full_box_like(x: jax.Array) -> jax.Array:
    t = jnp.ones_like(x[..., 0], dtype=jnp.asarray(x).dtype)
    inf = jnp.inf * t
    return acb_core.acb_box(di.interval(-inf, inf), di.interval(-inf, inf))


def interval_from_point(x: jax.Array) -> jax.Array:
    return di.interval(di._below(x), di._above(x))


def box_from_point(z: jax.Array) -> jax.Array:
    re = jnp.real(z)
    im = jnp.imag(z)
    return acb_core.acb_box(
        di.interval(di._below(re), di._above(re)),
        di.interval(di._below(im), di._above(im)),
    )


def interval_is_finite(x: jax.Array) -> jax.Array:
    return jnp.isfinite(x[..., 0]) & jnp.isfinite(x[..., 1])


def box_is_finite(x: jax.Array) -> jax.Array:
    re = acb_core.acb_real(x)
    im = acb_core.acb_imag(x)
    return interval_is_finite(re) & interval_is_finite(im)


def complex_is_finite(z: jax.Array) -> jax.Array:
    return jnp.isfinite(jnp.real(z)) & jnp.isfinite(jnp.imag(z))


def interval_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    lo = jnp.sum(xs[..., 0], axis=axis)
    hi = jnp.sum(xs[..., 1], axis=axis)
    return di.interval(di._below(lo), di._above(hi))


def box_sum(xs: jax.Array, axis: int = -1) -> jax.Array:
    re = acb_core.acb_real(xs)
    im = acb_core.acb_imag(xs)
    re_out = di.interval(di._below(jnp.sum(re[..., 0], axis=axis)), di._above(jnp.sum(re[..., 1], axis=axis)))
    im_out = di.interval(di._below(jnp.sum(im[..., 0], axis=axis)), di._above(jnp.sum(im[..., 1], axis=axis)))
    return acb_core.acb_box(re_out, im_out)


def interval_trace(a: jax.Array) -> jax.Array:
    n = a.shape[-2]
    idx = jnp.arange(n)
    return interval_sum(a[..., idx, idx, :], axis=-1)


def box_trace(a: jax.Array) -> jax.Array:
    n = a.shape[-2]
    idx = jnp.arange(n)
    return box_sum(a[..., idx, idx, :], axis=-1)


def interval_det_2x2(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    return di.fast_sub(di.fast_mul(a00, a11), di.fast_mul(a01, a10))


def box_det_2x2(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    return acb_core.acb_sub(acb_core.acb_mul(a00, a11), acb_core.acb_mul(a01, a10))


def interval_det_3x3(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a02 = a[..., 0, 2, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    a12 = a[..., 1, 2, :]
    a20 = a[..., 2, 0, :]
    a21 = a[..., 2, 1, :]
    a22 = a[..., 2, 2, :]
    pos = interval_sum(
        jnp.stack(
            [
                di.fast_mul(di.fast_mul(a00, a11), a22),
                di.fast_mul(di.fast_mul(a01, a12), a20),
                di.fast_mul(di.fast_mul(a02, a10), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    neg = interval_sum(
        jnp.stack(
            [
                di.fast_mul(di.fast_mul(a02, a11), a20),
                di.fast_mul(di.fast_mul(a01, a10), a22),
                di.fast_mul(di.fast_mul(a00, a12), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    return di.fast_sub(pos, neg)


def box_det_3x3(a: jax.Array) -> jax.Array:
    a00 = a[..., 0, 0, :]
    a01 = a[..., 0, 1, :]
    a02 = a[..., 0, 2, :]
    a10 = a[..., 1, 0, :]
    a11 = a[..., 1, 1, :]
    a12 = a[..., 1, 2, :]
    a20 = a[..., 2, 0, :]
    a21 = a[..., 2, 1, :]
    a22 = a[..., 2, 2, :]
    pos = box_sum(
        jnp.stack(
            [
                acb_core.acb_mul(acb_core.acb_mul(a00, a11), a22),
                acb_core.acb_mul(acb_core.acb_mul(a01, a12), a20),
                acb_core.acb_mul(acb_core.acb_mul(a02, a10), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    neg = box_sum(
        jnp.stack(
            [
                acb_core.acb_mul(acb_core.acb_mul(a02, a11), a20),
                acb_core.acb_mul(acb_core.acb_mul(a01, a10), a22),
                acb_core.acb_mul(acb_core.acb_mul(a00, a12), a21),
            ],
            axis=-2,
        ),
        axis=-1,
    )
    return acb_core.acb_sub(pos, neg)


__all__ = [
    "as_interval_mat_2x2",
    "as_interval_matrix",
    "as_interval_vector",
    "as_box_mat_2x2",
    "as_box_matrix",
    "as_box_vector",
    "full_interval_like",
    "full_box_like",
    "interval_from_point",
    "box_from_point",
    "interval_is_finite",
    "box_is_finite",
    "complex_is_finite",
    "interval_sum",
    "box_sum",
    "interval_trace",
    "box_trace",
    "interval_det_2x2",
    "box_det_2x2",
    "interval_det_3x3",
    "box_det_3x3",
]
