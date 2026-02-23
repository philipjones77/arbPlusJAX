from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def arb_fpwrap_double_exp(x: jax.Array) -> jax.Array:
    return jnp.exp(jnp.asarray(x, dtype=jnp.float64))


def arb_fpwrap_double_log(x: jax.Array) -> jax.Array:
    return jnp.log(jnp.asarray(x, dtype=jnp.float64))


def arb_fpwrap_cdouble_exp(x: jax.Array) -> jax.Array:
    return jnp.exp(jnp.asarray(x, dtype=jnp.complex128))


def arb_fpwrap_cdouble_log(x: jax.Array) -> jax.Array:
    return jnp.log(jnp.asarray(x, dtype=jnp.complex128))


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
