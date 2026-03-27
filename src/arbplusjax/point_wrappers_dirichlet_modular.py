from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from . import elementary as el
from . import point_wrappers_core as core


@partial(jax.jit, static_argnames=("n_terms",))
def acb_dirichlet_zeta_point(s: jax.Array, n_terms: int = 64) -> jax.Array:
    out = core._vectorize_complex_scalar(lambda ss: core._complex_zeta_scalar(ss, n_terms), s)
    return out.astype(el.complex_dtype_from(s))


@partial(jax.jit, static_argnames=("n_terms",))
def acb_dirichlet_eta_point(s: jax.Array, n_terms: int = 64) -> jax.Array:
    ss = el.as_complex(s)
    zeta = acb_dirichlet_zeta_point(ss, n_terms=n_terms)
    one = jnp.asarray(1.0, dtype=el.real_dtype_from_complex_dtype(ss.dtype))
    two = jnp.asarray(2.0, dtype=el.real_dtype_from_complex_dtype(ss.dtype))
    factor = one - jnp.exp((ss - one) * jnp.log(two))
    return (factor * zeta).astype(el.complex_dtype_from(s))


@partial(jax.jit, static_argnames=())
def acb_modular_j_point(tau: jax.Array) -> jax.Array:
    tt = el.as_complex(tau)
    real_dtype = el.real_dtype_from_complex_dtype(tt.dtype)
    q = jnp.exp(jnp.asarray(2j, dtype=tt.dtype) * jnp.asarray(el.PI, dtype=real_dtype) * tt)
    c744 = jnp.asarray(744.0, dtype=real_dtype)
    c1 = jnp.asarray(196884.0, dtype=real_dtype)
    c2 = jnp.asarray(21493760.0, dtype=real_dtype)
    return jnp.asarray(1.0, dtype=real_dtype) / q + c744 + c1 * q + c2 * q * q


def acb_dirichlet_zeta_batch_fixed_point(s: jax.Array, *, n_terms: int = 64) -> jax.Array:
    return acb_dirichlet_zeta_point(s, n_terms=n_terms)


def acb_dirichlet_zeta_batch_padded_point(s: jax.Array, *, pad_to: int, n_terms: int = 64) -> jax.Array:
    call_args, _ = core._pad_point_batch_last((s,), pad_to)
    return acb_dirichlet_zeta_point(*call_args, n_terms=n_terms)


def acb_dirichlet_eta_batch_fixed_point(s: jax.Array, *, n_terms: int = 64) -> jax.Array:
    return acb_dirichlet_eta_point(s, n_terms=n_terms)


def acb_dirichlet_eta_batch_padded_point(s: jax.Array, *, pad_to: int, n_terms: int = 64) -> jax.Array:
    call_args, _ = core._pad_point_batch_last((s,), pad_to)
    return acb_dirichlet_eta_point(*call_args, n_terms=n_terms)


def acb_modular_j_batch_fixed_point(tau: jax.Array) -> jax.Array:
    return acb_modular_j_point(tau)


def acb_modular_j_batch_padded_point(tau: jax.Array, *, pad_to: int) -> jax.Array:
    call_args, _ = core._pad_point_batch_last((tau,), pad_to)
    return acb_modular_j_point(*call_args)


__all__ = [
    "acb_dirichlet_zeta_point",
    "acb_dirichlet_eta_point",
    "acb_modular_j_point",
    "acb_dirichlet_zeta_batch_fixed_point",
    "acb_dirichlet_zeta_batch_padded_point",
    "acb_dirichlet_eta_batch_fixed_point",
    "acb_dirichlet_eta_batch_padded_point",
    "acb_modular_j_batch_fixed_point",
    "acb_modular_j_batch_padded_point",
]
