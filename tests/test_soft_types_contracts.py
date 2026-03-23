import jax
import jax.numpy as jnp

from arbplusjax.soft_types import SoftBool, SoftIndex


def test_soft_bool_clipping_hardening_and_pytree_round_trip():
    value = SoftBool(jnp.asarray(1.25, dtype=jnp.float32))
    leaves, treedef = jax.tree_util.tree_flatten(value)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jnp.isclose(value.clipped(), 1.0)
    assert bool(value.hard())
    assert isinstance(rebuilt, SoftBool)
    assert jnp.isclose(rebuilt.prob, value.prob)


def test_soft_index_normalization_hard_selection_and_expectation():
    value = SoftIndex(jnp.asarray([1.0, 3.0, 2.0], dtype=jnp.float32))

    normalized = value.normalized()
    assert jnp.isclose(jnp.sum(normalized), 1.0)
    assert int(value.hard()) == 1
    assert jnp.isclose(value.expectation(), (0.0 * 1.0 + 1.0 * 3.0 + 2.0 * 2.0) / 6.0)
