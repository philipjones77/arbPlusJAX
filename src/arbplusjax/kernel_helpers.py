from __future__ import annotations

import jax
import jax.numpy as jnp

from . import acb_core
from . import double_interval as di


def batch_size(args: tuple[object, ...]) -> int:
    if not args:
        raise ValueError("batch call requires at least one argument array")
    n: int | None = None
    for arg in args:
        if not hasattr(arg, "ndim") or not hasattr(arg, "shape"):
            raise ValueError("batch call requires array arguments")
        if arg.ndim == 0:
            raise ValueError("batch call requires arrays with a leading batch dimension")
        cur = int(arg.shape[0])
        if n is None:
            n = cur
        elif cur != n:
            raise ValueError("all batch arguments must share the same leading dimension")
    assert n is not None
    return n


def mixed_batch_size_or_none(args: tuple[object, ...]) -> int | None:
    batch_n: int | None = None
    for arg in args:
        if hasattr(arg, "ndim") and hasattr(arg, "shape") and getattr(arg, "ndim") > 0:
            cur = int(arg.shape[0])
            if batch_n is None:
                batch_n = cur
            elif cur != batch_n:
                raise ValueError("all batch arguments must share the same leading dimension")
    return batch_n


def pad_batch_args(
    args: tuple[object, ...],
    *,
    pad_to: int | None,
    pad_value: float | complex = 0.0,
) -> tuple[tuple[object, ...], int]:
    n = batch_size(args)
    if pad_to is None:
        return args, n
    target = int(pad_to)
    if target < n:
        raise ValueError(f"pad_to must be >= batch size; got pad_to={target}, batch={n}")
    if target == n:
        return args, n

    padded: list[object] = []
    for arg in args:
        arr = jnp.asarray(arg)
        pad_shape = (target - n,) + tuple(arr.shape[1:])
        pad_block = jnp.full(pad_shape, pad_value, dtype=arr.dtype)
        padded.append(jnp.concatenate((arr, pad_block), axis=0))
    return tuple(padded), n


def pad_mixed_batch_args_repeat_last(args: tuple[object, ...], *, pad_to: int | None) -> tuple[tuple[object, ...], int]:
    batch_n = mixed_batch_size_or_none(args)
    if batch_n is None:
        raise ValueError("batch fastpath requires at least one array argument")
    if pad_to is None or int(pad_to) == batch_n:
        return args, batch_n
    target = int(pad_to)
    if target < batch_n:
        raise ValueError(f"pad_to must be >= batch size; got pad_to={target}, batch={batch_n}")

    padded: list[object] = []
    for arg in args:
        if hasattr(arg, "ndim") and hasattr(arg, "shape") and getattr(arg, "ndim") > 0:
            arr = jnp.asarray(arg)
            pad_row = arr[-1:, ...] if batch_n > 0 else jnp.zeros((1,) + tuple(arr.shape[1:]), dtype=arr.dtype)
            pad_block = jnp.repeat(pad_row, target - batch_n, axis=0)
            padded.append(jnp.concatenate((arr, pad_block), axis=0))
        else:
            padded.append(arg)
    return tuple(padded), batch_n


def trim_batch_out(out: object, n: int):
    if isinstance(out, tuple):
        return tuple(trim_batch_out(item, n) for item in out)
    if hasattr(out, "ndim") and out.ndim > 0:
        return out[:n]
    return out


def midpoint_from_interval_like(out: object):
    if isinstance(out, tuple):
        return tuple(midpoint_from_interval_like(item) for item in out)
    arr = jnp.asarray(out)
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return di.midpoint(arr)
    if arr.ndim >= 1 and arr.shape[-1] == 4:
        return acb_core.acb_midpoint(arr)
    return out


def point_interval(x: jax.Array) -> jax.Array:
    arr = jnp.asarray(x)
    return di.interval(arr, arr)


def point_box(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z)
    return acb_core.acb_box(di.interval(jnp.real(zz), jnp.real(zz)), di.interval(jnp.imag(zz), jnp.imag(zz)))


def scalarize_unary_complex(fn):
    @jax.jit
    def wrapped(z: jax.Array) -> jax.Array:
        flat = jnp.ravel(jnp.asarray(z, dtype=jnp.complex128))
        out = jax.vmap(lambda t: acb_core.acb_midpoint(fn(point_box(t))))(flat)
        return out.reshape(jnp.shape(z))

    return wrapped


def scalarize_binary_complex(fn):
    @jax.jit
    def wrapped(x: jax.Array, y: jax.Array) -> jax.Array:
        xx = jnp.asarray(x, dtype=jnp.complex128)
        yy = jnp.asarray(y, dtype=jnp.complex128)
        flat_x = jnp.ravel(xx)
        flat_y = jnp.ravel(yy)
        out = jax.vmap(lambda a, b: acb_core.acb_midpoint(fn(point_box(a), point_box(b))))(flat_x, flat_y)
        return out.reshape(jnp.shape(xx))

    return wrapped


def vmap_complex_scalar(fn):
    @jax.jit
    def wrapped(z: jax.Array) -> jax.Array:
        zz = jnp.asarray(z, dtype=jnp.complex128)
        flat = jnp.ravel(zz)
        out = jax.vmap(fn)(flat)
        return out.reshape(jnp.shape(zz))

    return wrapped

