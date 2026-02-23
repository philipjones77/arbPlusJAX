from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def bernoulli_number(n: jax.Array) -> jax.Array:
    n = jnp.asarray(n, dtype=jnp.int64)
    return jnp.where(
        n == 0,
        1.0,
        jnp.where(
            n == 1,
            -0.5,
            jnp.where(
                n == 2,
                1.0 / 6.0,
                jnp.where(n == 4, -1.0 / 30.0, 0.0),
            ),
        ),
    ).astype(jnp.float64)


def bernoulli_number_batch(n: jax.Array) -> jax.Array:
    return bernoulli_number(n)


bernoulli_number_batch_jit = jax.jit(bernoulli_number_batch)


__all__ = [
    "bernoulli_number",
    "bernoulli_number_batch",
    "bernoulli_number_batch_jit",
]
