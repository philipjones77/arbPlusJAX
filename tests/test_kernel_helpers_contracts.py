from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import kernel_helpers as kh


def test_kernel_helpers_batch_padding_and_trimming_contracts() -> None:
    x = jnp.asarray([1.0, 2.0, 3.0])
    y = jnp.asarray([[10.0], [20.0], [30.0]])

    assert kh.batch_size((x, y)) == 3
    assert kh.mixed_batch_size_or_none((x, 5.0, y)) == 3

    padded, original_n = kh.pad_batch_args((x, y), pad_to=5, pad_value=-1.0)
    assert original_n == 3
    assert padded[0].shape == (5,)
    assert padded[1].shape == (5, 1)
    assert jnp.array_equal(padded[0][-2:], jnp.asarray([-1.0, -1.0]))

    mixed_padded, mixed_n = kh.pad_mixed_batch_args_repeat_last((x, 5.0, y), pad_to=5)
    assert mixed_n == 3
    assert jnp.array_equal(mixed_padded[0][-2:], jnp.asarray([3.0, 3.0]))
    assert mixed_padded[1] == 5.0
    assert jnp.array_equal(mixed_padded[2][-2:, 0], jnp.asarray([30.0, 30.0]))

    trimmed = kh.trim_batch_out((padded[0], {"ignored": True}, padded[1]), 3)
    assert jnp.array_equal(trimmed[0], x)
    assert trimmed[1] == {"ignored": True}
    assert jnp.array_equal(trimmed[2], y)

    with pytest.raises(ValueError):
        kh.batch_size(())
    with pytest.raises(ValueError):
        kh.batch_size((jnp.asarray(2.0),))
    with pytest.raises(ValueError):
        kh.pad_batch_args((x,), pad_to=2)


def test_kernel_helpers_interval_box_and_scalarization_contracts() -> None:
    iv = di.interval(jnp.asarray([1.0, 2.0]), jnp.asarray([1.5, 2.5]))
    box = acb_core.acb_box(iv, di.interval(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.5])))

    midpoint = kh.midpoint_from_interval_like((iv, box))
    assert jnp.allclose(midpoint[0], jnp.asarray([1.25, 2.25]))
    assert jnp.allclose(midpoint[1], jnp.asarray([1.25 + 0.0j, 2.25 + 1.25j]))

    point_iv = kh.point_interval(jnp.asarray([2.0, 3.0]))
    point_box = kh.point_box(jnp.asarray([2.0 + 1.0j, 3.0 - 2.0j]))
    assert jnp.array_equal(point_iv[..., 0], jnp.asarray([2.0, 3.0]))
    assert jnp.array_equal(point_iv[..., 1], jnp.asarray([2.0, 3.0]))
    assert jnp.array_equal(acb_core.acb_real(point_box), di.interval(jnp.asarray([2.0, 3.0]), jnp.asarray([2.0, 3.0])))

    unary = kh.scalarize_unary_complex(lambda z: z * (1.0 + 1.0j))
    binary = kh.scalarize_binary_complex(lambda x, y: x + 2.0 * y)
    vmapped = kh.vmap_complex_scalar(lambda z: jnp.conj(z) + 1.0j)

    z = jnp.asarray([[1.0 + 2.0j, 3.0 - 1.0j]])
    w = jnp.asarray([[0.5 - 1.0j, -2.0 + 0.25j]])
    assert jnp.allclose(unary(z), z * (1.0 + 1.0j))
    assert jnp.allclose(binary(z, w), z + 2.0 * w)
    assert jnp.allclose(vmapped(z), jnp.conj(z) + 1.0j)
