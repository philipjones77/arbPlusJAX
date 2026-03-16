import jax
import jax.numpy as jnp

from arbplusjax.soft_ops import (
    grad_replace,
    soft_argmax,
    soft_clip,
    soft_heaviside,
    soft_sign,
    soft_take_along_axis,
    soft_where,
    st,
)
from arbplusjax.soft_types import SoftBool, SoftIndex

from tests._test_checks import _check


def test_st_and_grad_replace_use_soft_backward_values():
    x = jnp.asarray(2.0, dtype=jnp.float64)

    g_st = jax.grad(lambda t: st(jnp.round(t), t * t))(x)
    g_replace = jax.grad(lambda t: grad_replace(jnp.sign(t), 3.0 * t))(x)

    _check(bool(jnp.allclose(g_st, 4.0, rtol=0.0, atol=0.0)))
    _check(bool(jnp.allclose(g_replace, 3.0, rtol=0.0, atol=0.0)))


def test_soft_bool_and_where_are_probability_weighted():
    sb = soft_heaviside(jnp.asarray([-2.0, 2.0], dtype=jnp.float64), temperature=0.5)
    out = soft_where(sb, jnp.asarray([10.0, 10.0]), jnp.asarray([0.0, 0.0]))

    _check(isinstance(sb, SoftBool))
    _check(bool(jnp.all(sb.clipped() >= 0.0)))
    _check(bool(jnp.all(sb.clipped() <= 1.0)))
    _check(bool(out[0] < out[1]))


def test_soft_sign_and_clip_shapes_and_ranges():
    x = jnp.asarray([-2.0, -0.1, 0.1, 2.0], dtype=jnp.float64)
    s = soft_sign(x, temperature=0.5)
    c = soft_clip(x, -0.5, 0.5, temperature=0.25)

    _check(s.shape == x.shape)
    _check(c.shape == x.shape)
    _check(bool(jnp.all(jnp.abs(s) <= 1.0 + 1e-12)))
    _check(bool(jnp.all(c >= -0.55)))
    _check(bool(jnp.all(c <= 0.55)))


def test_soft_index_argmax_and_take_along_axis():
    scores = jnp.asarray([0.0, 1.0, 4.0], dtype=jnp.float64)
    vals = jnp.asarray([3.0, 5.0, 11.0], dtype=jnp.float64)
    idx = soft_argmax(scores, temperature=0.2)
    taken = soft_take_along_axis(vals, idx)

    _check(isinstance(idx, SoftIndex))
    _check(bool(jnp.allclose(jnp.sum(idx.normalized()), 1.0, rtol=1e-12, atol=1e-12)))
    _check(int(idx.hard()) == 2)
    _check(bool(taken > 8.0))
