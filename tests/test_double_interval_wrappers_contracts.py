from __future__ import annotations

import jax.numpy as jnp
import pytest

from arbplusjax import double_interval as di
from arbplusjax import double_interval_wrappers as diw


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def test_double_interval_wrappers_modes_cover_basic_rigorous_and_adaptive() -> None:
    x = di.interval(jnp.asarray([2.0]), jnp.asarray([3.0]))
    y = di.interval(jnp.asarray([4.0]), jnp.asarray([5.0]))

    add_basic = diw.fast_add_mode(x, y, impl="basic")
    add_baseline = diw.fast_add_mode(x, y, impl="baseline")
    add_rigorous = diw.fast_add_mode(x, y, impl="rigorous", prec_bits=80)
    add_adaptive = diw.fast_add_mode(x, y, impl="adaptive", prec_bits=80)
    assert jnp.array_equal(add_basic, add_baseline)
    assert _contains_interval(add_rigorous, add_basic)
    assert _contains_interval(add_adaptive, add_rigorous)

    mul_basic = diw.fast_mul_mode(x, y, impl="basic")
    mul_rigorous = diw.fast_mul_mode(x, y, impl="rigorous", prec_bits=80)
    sqr_adaptive = diw.fast_sqr_mode(x, impl="adaptive", prec_bits=80)
    log_basic = diw.fast_log_nonnegative_mode(x, impl="basic")
    div_basic = diw.fast_div_mode(y, x, impl="basic")

    assert mul_basic.shape == (1, 2)
    assert _contains_interval(mul_rigorous, mul_basic)
    assert sqr_adaptive.shape == (1, 2)
    assert log_basic.shape == (1, 2)
    assert div_basic.shape == (1, 2)


def test_double_interval_wrappers_reject_invalid_mode() -> None:
    x = di.interval(jnp.asarray([1.0]), jnp.asarray([2.0]))
    with pytest.raises(ValueError):
        diw.fast_sub_mode(x, x, impl="bad")
