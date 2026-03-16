from __future__ import annotations

import jax
import jax.numpy as jnp

from .soft_types import SoftBool, SoftIndex


def grad_replace(forward_value: jax.Array, backward_value: jax.Array) -> jax.Array:
    """Use `forward_value` in the primal pass and `backward_value` for gradients."""
    fwd = jnp.asarray(forward_value)
    bwd = jnp.asarray(backward_value, dtype=fwd.dtype)
    return lax_stop(fwd) + (bwd - lax_stop(bwd))


def st(hard_value: jax.Array, soft_value: jax.Array) -> jax.Array:
    """Straight-through estimator: hard forward pass, soft backward pass."""
    return grad_replace(hard_value, soft_value)


def lax_stop(x: jax.Array) -> jax.Array:
    return jax.lax.stop_gradient(jnp.asarray(x))


def soft_sign(x: jax.Array, *, temperature: float = 1.0, straight_through: bool = False) -> jax.Array:
    xx = jnp.asarray(x)
    scale = jnp.asarray(temperature, dtype=xx.dtype)
    soft = jnp.tanh(xx / jnp.maximum(scale, jnp.asarray(1e-12, dtype=xx.dtype)))
    if straight_through:
        return st(jnp.sign(xx), soft)
    return soft


def soft_heaviside(x: jax.Array, *, temperature: float = 1.0, straight_through: bool = False) -> SoftBool:
    xx = jnp.asarray(x)
    scale = jnp.asarray(temperature, dtype=xx.dtype)
    prob = jax.nn.sigmoid(xx / jnp.maximum(scale, jnp.asarray(1e-12, dtype=xx.dtype)))
    if straight_through:
        hard = jnp.where(xx >= 0.0, jnp.asarray(1.0, dtype=xx.dtype), jnp.asarray(0.0, dtype=xx.dtype))
        prob = st(hard, prob)
    return SoftBool(prob=prob)


def soft_clip(
    x: jax.Array,
    a_min: jax.Array,
    a_max: jax.Array,
    *,
    temperature: float = 1.0,
    straight_through: bool = False,
) -> jax.Array:
    xx = jnp.asarray(x)
    lo = jnp.asarray(a_min, dtype=xx.dtype)
    hi = jnp.asarray(a_max, dtype=xx.dtype)
    scale = jnp.maximum(jnp.asarray(temperature, dtype=xx.dtype), jnp.asarray(1e-12, dtype=xx.dtype))
    softplus = lambda t: scale * jax.nn.softplus(t / scale)
    soft = lo + softplus(xx - lo) - softplus(xx - hi)
    if straight_through:
        return st(jnp.clip(xx, lo, hi), soft)
    return soft


def soft_where(condition: SoftBool | jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    yy = jnp.asarray(y, dtype=xx.dtype)
    if isinstance(condition, SoftBool):
        w = condition.clipped().astype(xx.dtype)
    else:
        w = jnp.asarray(condition, dtype=xx.dtype)
    return w * xx + (1.0 - w) * yy


def soft_argmax(x: jax.Array, *, temperature: float = 1.0, axis: int = -1, straight_through: bool = False) -> SoftIndex:
    xx = jnp.asarray(x)
    scale = jnp.maximum(jnp.asarray(temperature, dtype=xx.dtype), jnp.asarray(1e-12, dtype=xx.dtype))
    probs = jax.nn.softmax(xx / scale, axis=axis)
    if straight_through:
        hard_idx = jnp.argmax(xx, axis=axis)
        hard = jax.nn.one_hot(hard_idx, xx.shape[axis], dtype=xx.dtype)
        probs = st(hard, probs)
    return SoftIndex(probs=probs)


def soft_take_along_axis(arr: jax.Array, index: SoftIndex, *, axis: int = -1) -> jax.Array:
    aa = jnp.asarray(arr)
    probs = index.normalized().astype(aa.dtype)
    if axis != -1:
        probs = jnp.moveaxis(probs, -1, axis)
    return jnp.sum(aa * probs, axis=axis)


def soft_top_k(x: jax.Array, k: int, *, temperature: float = 1.0) -> SoftIndex:
    xx = jnp.asarray(x)
    logits = xx / jnp.maximum(jnp.asarray(temperature, dtype=xx.dtype), jnp.asarray(1e-12, dtype=xx.dtype))
    probs = jax.nn.softmax(logits, axis=-1)
    top_idx = jnp.argsort(probs)[-k:]
    mask = jnp.zeros_like(probs).at[top_idx].set(1.0)
    return SoftIndex(probs=probs * mask)


__all__ = [
    "grad_replace",
    "st",
    "soft_sign",
    "soft_heaviside",
    "soft_clip",
    "soft_where",
    "soft_argmax",
    "soft_take_along_axis",
    "soft_top_k",
]
