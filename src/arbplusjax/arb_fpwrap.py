from __future__ import annotations

import jax
import jax.numpy as jnp



def arb_fpwrap_double_exp(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if not jnp.issubdtype(xx.dtype, jnp.floating):
        xx = jnp.asarray(xx, dtype=jnp.float64)
    return jnp.exp(xx)


def arb_fpwrap_double_log(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if not jnp.issubdtype(xx.dtype, jnp.floating):
        xx = jnp.asarray(xx, dtype=jnp.float64)
    return jnp.log(xx)


def arb_fpwrap_cdouble_exp(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        zz = xx
    elif jnp.issubdtype(xx.dtype, jnp.floating):
        zz = xx.astype(jnp.complex64 if xx.dtype == jnp.float32 else jnp.complex128)
    else:
        zz = jnp.asarray(xx, dtype=jnp.complex128)
    return jnp.exp(zz)


def arb_fpwrap_cdouble_log(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    if jnp.issubdtype(xx.dtype, jnp.complexfloating):
        zz = xx
    elif jnp.issubdtype(xx.dtype, jnp.floating):
        zz = xx.astype(jnp.complex64 if xx.dtype == jnp.float32 else jnp.complex128)
    else:
        zz = jnp.asarray(xx, dtype=jnp.complex128)
    return jnp.log(zz)


arb_fpwrap_double_exp_jit = jax.jit(arb_fpwrap_double_exp)
arb_fpwrap_double_log_jit = jax.jit(arb_fpwrap_double_log)
arb_fpwrap_cdouble_exp_jit = jax.jit(arb_fpwrap_cdouble_exp)
arb_fpwrap_cdouble_log_jit = jax.jit(arb_fpwrap_cdouble_log)


__all__ = [
    "arb_fpwrap_double_exp",
    "arb_fpwrap_double_log",
    "arb_fpwrap_cdouble_exp",
    "arb_fpwrap_cdouble_log",
    "arb_fpwrap_double_exp_jit",
    "arb_fpwrap_double_log_jit",
    "arb_fpwrap_cdouble_exp_jit",
    "arb_fpwrap_cdouble_log_jit",
]
