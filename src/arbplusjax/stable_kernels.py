from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from . import acb_core
from . import api
from . import double_interval as di
from .capability_registry import DOWNSTREAM_KERNELS, lookup_capability


_MODE_SET = ("point", "basic", "adaptive", "rigorous")


def _check_mode(mode: str) -> str:
    if mode not in _MODE_SET:
        raise ValueError(f"unsupported mode {mode!r}; expected one of {_MODE_SET}")
    return mode


def _eval_unary(name: str, x: jax.Array, *, mode: str, dtype: str | jnp.dtype | None, **kwargs) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return api.eval_point(name, x, dtype=dtype, **kwargs)
    return api.eval_interval(name, x, mode=mode, dtype=dtype, **kwargs)


def _cast_dtype(arg: jax.Array, dtype: str | jnp.dtype | None) -> jax.Array:
    arr = jnp.asarray(arg)
    if dtype is None:
        return arr
    return arr.astype(jnp.dtype(dtype))


def _loggamma_name(x: jax.Array) -> str:
    dt = jnp.asarray(x).dtype
    return "acb_lgamma" if jnp.issubdtype(dt, jnp.complexfloating) else "arb_lgamma"


def _gamma_name(x: jax.Array) -> str:
    dt = jnp.asarray(x).dtype
    return "acb_gamma" if jnp.issubdtype(dt, jnp.complexfloating) else "arb_gamma"


def _box_scalar(z: jax.Array) -> jax.Array:
    zz = jnp.asarray(z)
    return acb_core.acb_box(di.interval(jnp.real(zz), jnp.real(zz)), di.interval(jnp.imag(zz), jnp.imag(zz)))


def _bound_input(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        return _box_scalar(xx)
    return di.interval(xx, xx)


def _eval_unary_batch(
    name: str,
    x: jax.Array,
    *,
    mode: str,
    dtype: str | jnp.dtype | None,
    pad_to: int | None,
    **kwargs,
) -> jax.Array:
    mode = _check_mode(mode)
    if mode == "point":
        return api.eval_point_batch(name, x, dtype=dtype, pad_to=pad_to, **kwargs)
    return api.eval_interval_batch(name, x, mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)


def gamma(x: jax.Array, *, mode: str = "point", dtype: str | jnp.dtype | None = None, **kwargs) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    if _check_mode(mode) == "point":
        return api.eval_point("gamma", cast, **kwargs)
    return api.eval_interval(_gamma_name(cast), _bound_input(cast), mode=mode, **kwargs)


def gamma_batch(
    x: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    return _eval_unary_batch("gamma", x, mode=mode, dtype=dtype, pad_to=pad_to, **kwargs)


def loggamma(x: jax.Array, *, mode: str = "point", dtype: str | jnp.dtype | None = None, **kwargs) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    if _check_mode(mode) == "point":
        return _eval_unary(_loggamma_name(cast), cast, mode=mode, dtype=None, **kwargs)
    return api.eval_interval(_loggamma_name(cast), _bound_input(cast), mode=mode, **kwargs)


def loggamma_batch(
    x: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    pad_to: int | None = None,
    **kwargs,
) -> jax.Array:
    cast = _cast_dtype(x, dtype)
    return _eval_unary_batch(_loggamma_name(cast), cast, mode=mode, dtype=None, pad_to=pad_to, **kwargs)


def incomplete_gamma_lower(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_lower(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_gamma_lower_batch(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_lower_batch(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_gamma_upper(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_upper(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_gamma_upper_batch(
    s: jax.Array,
    z: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_gamma_upper_batch(_cast_dtype(s, dtype), _cast_dtype(z, dtype), mode=_check_mode(mode), **kwargs)


def incomplete_bessel_i(
    nu: jax.Array,
    z: jax.Array,
    upper_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_i(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(upper_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def incomplete_bessel_i_batch(
    nu: jax.Array,
    z: jax.Array,
    upper_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_i_batch(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(upper_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def incomplete_bessel_k(
    nu: jax.Array,
    z: jax.Array,
    lower_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_k(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(lower_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def incomplete_bessel_k_batch(
    nu: jax.Array,
    z: jax.Array,
    lower_limit: jax.Array,
    *,
    mode: str = "point",
    dtype: str | jnp.dtype | None = None,
    **kwargs,
) -> jax.Array:
    return api.incomplete_bessel_k_batch(
        _cast_dtype(nu, dtype),
        _cast_dtype(z, dtype),
        _cast_dtype(lower_limit, dtype),
        mode=_check_mode(mode),
        **kwargs,
    )


def list_supported_kernels() -> tuple[str, ...]:
    return tuple(sorted(DOWNSTREAM_KERNELS))


def get_kernel_capability(name: str) -> dict[str, object]:
    return lookup_capability(name)


__all__ = [
    "gamma",
    "gamma_batch",
    "loggamma",
    "loggamma_batch",
    "incomplete_gamma_lower",
    "incomplete_gamma_lower_batch",
    "incomplete_gamma_upper",
    "incomplete_gamma_upper_batch",
    "incomplete_bessel_i",
    "incomplete_bessel_i_batch",
    "incomplete_bessel_k",
    "incomplete_bessel_k_batch",
    "list_supported_kernels",
    "get_kernel_capability",
]
