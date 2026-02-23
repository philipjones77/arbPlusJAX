from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _partitions_table_fixed(n_max: int) -> jax.Array:
    n_max = int(max(n_max, 0))
    p = jnp.zeros((n_max + 1,), dtype=jnp.int64)
    p = p.at[0].set(1)

    def body_k(k, p_state):
        def body_m(m, acc):
            g1 = (m * (3 * m - 1)) // 2
            g2 = (m * (3 * m + 1)) // 2
            term1 = jnp.where(g1 <= k, p_state[k - g1], 0)
            term2 = jnp.where(g2 <= k, p_state[k - g2], 0)
            sign = jnp.where(m & 1, 1, -1)
            return acc + sign * (term1 + term2)

        acc = lax.fori_loop(1, k + 1, body_m, jnp.int64(0))
        return p_state.at[k].set(acc)

    p = lax.fori_loop(1, n_max + 1, body_k, p)
    return p


def partitions_p(n: jax.Array) -> jax.Array:
    n = jnp.asarray(n, dtype=jnp.int64)
    n = jnp.maximum(n, 0)
    p = _partitions_table_fixed(int(n))
    return p[n]

@partial(jax.jit, static_argnames=("n_max",))
def _partitions_p_batch_fixed(n: jax.Array, n_max: int) -> jax.Array:
    n = jnp.asarray(n, dtype=jnp.int64)
    n = jnp.maximum(n, 0)
    p = _partitions_table_fixed(n_max)
    return p[n]


def partitions_p_batch(n: jax.Array) -> jax.Array:
    n = jnp.asarray(n, dtype=jnp.int64)
    n = jnp.maximum(n, 0)
    n_max = int(jnp.max(n))
    return _partitions_p_batch_fixed(n, n_max)


partitions_p_batch_jit = partitions_p_batch


__all__ = [
    "partitions_p",
    "partitions_p_batch",
    "partitions_p_batch_jit",
]
