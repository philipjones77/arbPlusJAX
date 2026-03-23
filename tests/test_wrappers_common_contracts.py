from __future__ import annotations

import jax.numpy as jnp
import pytest

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import precision
from arbplusjax import wrappers_common as wc


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def test_wrappers_common_precision_resolution_and_inflation_contracts() -> None:
    old_bits = precision.get_prec_bits()
    try:
        precision.set_prec_bits(77)
        assert wc.resolve_prec_bits(None, None) == 77
        assert wc.resolve_prec_bits(20, None) == precision.dps_to_bits(20)
        assert wc.resolve_prec_bits(20, 99) == 99

        x = di.interval(jnp.asarray([1.0]), jnp.asarray([1.5]))
        inflated = wc.inflate_interval(x, prec_bits=64, adaptive=False)
        adaptive = wc.inflate_interval(x, prec_bits=64, adaptive=True)
        assert _contains_interval(inflated, x)
        assert _contains_interval(adaptive, inflated)

        box = acb_core.acb_box(x, x)
        inflated_box = wc.inflate_acb(box, prec_bits=64, adaptive=True)
        assert _contains_interval(acb_core.acb_real(inflated_box), acb_core.acb_real(box))
        assert _contains_interval(acb_core.acb_imag(inflated_box), acb_core.acb_imag(box))
    finally:
        precision.set_prec_bits(old_bits)


def test_wrappers_common_dispatch_and_kernel_helpers_preserve_basic_enclosures() -> None:
    x = di.interval(jnp.asarray([1.0]), jnp.asarray([2.0]))

    def base_fn(arg, *, prec_bits):
        del prec_bits
        return di.fast_add(arg, di.interval(1.0, 1.0))

    def rig_fn(arg, *, prec_bits):
        del prec_bits
        return di.fast_add(arg, di.interval(1.25, 1.25))

    basic = wc.dispatch_mode("basic", None, base_fn, rig_fn, None, False, 53, (x,), {})
    rigorous = wc.dispatch_mode("rigorous", None, base_fn, rig_fn, None, False, 53, (x,), {})
    adaptive = wc.dispatch_mode("adaptive", None, base_fn, None, None, False, 53, (x,), {})
    baseline = wc.dispatch_mode("baseline", None, base_fn, rig_fn, None, False, 53, (x,), {})

    assert jnp.array_equal(basic, baseline)
    assert _contains_interval(rigorous, basic)
    assert _contains_interval(adaptive, basic)

    point = wc.dispatch_mode(
        "point",
        lambda arg: arg + 1.0,
        base_fn,
        rig_fn,
        None,
        False,
        53,
        (jnp.asarray([2.0]),),
        {},
    )
    assert jnp.array_equal(point, jnp.asarray([3.0]))

    rigorous_kernel = wc.rigorous_interval_kernel(lambda arg: di.fast_mul(arg, arg), (x,), 53)
    adaptive_kernel = wc.adaptive_interval_kernel(lambda arg: di.fast_mul(arg, arg), (x,), 53)
    assert _contains_interval(rigorous_kernel, di.fast_mul(di.interval(1.5, 1.5), di.interval(1.5, 1.5)))
    assert _contains_interval(adaptive_kernel, di.fast_mul(di.interval(1.5, 1.5), di.interval(1.5, 1.5)))

    with pytest.raises(ValueError):
        wc.dispatch_mode("bad", None, base_fn, rig_fn, None, False, 53, (x,), {})
