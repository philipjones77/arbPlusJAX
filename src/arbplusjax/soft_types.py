from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SoftBool:
    """Differentiable boolean-like value represented by a probability in [0, 1]."""

    prob: jax.Array

    def tree_flatten(self):
        return (jnp.asarray(self.prob),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (prob,) = children
        return cls(prob=prob)

    def clipped(self) -> jax.Array:
        return jnp.clip(jnp.asarray(self.prob), 0.0, 1.0)

    def hard(self, threshold: float = 0.5) -> jax.Array:
        return self.clipped() >= jnp.asarray(threshold, dtype=jnp.asarray(self.prob).dtype)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SoftIndex:
    """Differentiable index-like value represented by a probability distribution."""

    probs: jax.Array

    def tree_flatten(self):
        return (jnp.asarray(self.probs),), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (probs,) = children
        return cls(probs=probs)

    def normalized(self) -> jax.Array:
        probs = jnp.asarray(self.probs)
        total = jnp.sum(probs)
        safe_total = jnp.where(total > 0.0, total, jnp.asarray(1.0, dtype=probs.dtype))
        return probs / safe_total

    def hard(self) -> jax.Array:
        return jnp.argmax(self.normalized(), axis=-1)

    def expectation(self) -> jax.Array:
        probs = self.normalized()
        idx = jnp.arange(probs.shape[-1], dtype=probs.dtype)
        return jnp.sum(probs * idx, axis=-1)


__all__ = ["SoftBool", "SoftIndex"]
