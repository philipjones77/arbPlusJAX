from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import calc_wrappers as clw
from arbplusjax import double_interval as di


def _contains_interval(outer, inner) -> bool:
    return bool(jnp.all(outer[..., 0] <= inner[..., 0]) and jnp.all(outer[..., 1] >= inner[..., 1]))


def _contains_box(outer, inner) -> bool:
    return _contains_interval(acb_core.acb_real(outer), acb_core.acb_real(inner)) and _contains_interval(
        acb_core.acb_imag(outer), acb_core.acb_imag(inner)
    )


def test_calc_wrappers_real_and_complex_modes_return_expected_container_shapes() -> None:
    x = di.interval(jnp.float64(0.0), jnp.float64(1.0))
    y = di.interval(jnp.float64(1.0), jnp.float64(2.0))
    z0 = acb_core.acb_box(x, di.interval(jnp.float64(0.0), jnp.float64(0.0)))
    z1 = acb_core.acb_box(y, di.interval(jnp.float64(0.0), jnp.float64(0.0)))

    partition_basic = clw.arb_calc_partition_mode(x, y, parts=4, impl="basic")
    partition_rigorous = clw.arb_calc_partition_mode(x, y, parts=4, impl="rigorous")
    line_basic = clw.arb_calc_integrate_line_mode(x, y, integrand="exp", n_steps=16, impl="basic")
    line_adaptive = clw.arb_calc_integrate_line_mode(x, y, integrand="exp", n_steps=16, impl="adaptive")
    cauchy = clw.acb_calc_cauchy_bound_mode(z0, z1, integrand="exp", n_steps=16, impl="basic", prec_bits=80)
    complex_line_basic = clw.acb_calc_integrate_line_mode(z0, z1, integrand="exp", n_steps=16, impl="basic", prec_bits=80)
    complex_line_adaptive = clw.acb_calc_integrate_line_mode(z0, z1, integrand="exp", n_steps=16, impl="adaptive", prec_bits=80)

    assert partition_basic.shape[-1] == 2
    assert _contains_interval(partition_rigorous, partition_basic)
    assert line_basic.shape[-1] == 2
    assert _contains_interval(line_adaptive, line_basic)
    assert cauchy.shape[-1] == 2
    assert complex_line_basic.shape[-1] == 4
    assert _contains_box(complex_line_adaptive, complex_line_basic)
