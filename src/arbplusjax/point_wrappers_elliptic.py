from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import acb_elliptic
from . import elementary as el
from . import point_wrappers_core as core


@partial(jax.jit, static_argnames=())
def acb_elliptic_k_point(m: jax.Array) -> jax.Array:
    return core._vectorize_complex_scalar(
        lambda mm: jnp.asarray(el.HALF_PI, dtype=el.real_dtype_from_complex_dtype(el.as_complex(mm).dtype))
        / acb_elliptic._agm(
            jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(mm).dtype),
            jnp.sqrt(1.0 - el.as_complex(mm)),
            iters=8,
        ),
        m,
    )


@partial(jax.jit, static_argnames=())
def acb_elliptic_e_point(m: jax.Array) -> jax.Array:
    return core._vectorize_complex_scalar(
        lambda mm: jnp.asarray(el.HALF_PI, dtype=el.real_dtype_from_complex_dtype(el.as_complex(mm).dtype))
        * acb_elliptic._agm(
            jnp.asarray(1.0 + 0.0j, dtype=el.as_complex(mm).dtype),
            jnp.sqrt(1.0 - el.as_complex(mm)),
            iters=8,
        ),
        m,
    )


def acb_elliptic_k_batch_fixed_point(m: jax.Array) -> jax.Array:
    return acb_elliptic_k_point(m)


def acb_elliptic_k_batch_padded_point(m: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = core._pad_point_batch_last((m,), pad_to)
    return acb_elliptic_k_point(*call_args)


def acb_elliptic_e_batch_fixed_point(m: jax.Array) -> jax.Array:
    return acb_elliptic_e_point(m)


def acb_elliptic_e_batch_padded_point(m: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = core._pad_point_batch_last((m,), pad_to)
    return acb_elliptic_e_point(*call_args)


__all__ = [
    "acb_elliptic_k_point",
    "acb_elliptic_e_point",
    "acb_elliptic_k_batch_fixed_point",
    "acb_elliptic_k_batch_padded_point",
    "acb_elliptic_e_batch_fixed_point",
    "acb_elliptic_e_batch_padded_point",
]
