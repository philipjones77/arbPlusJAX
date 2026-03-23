from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import precision


def test_precision_global_setters_and_contexts_restore_state() -> None:
    old_dps = precision.get_dps()
    old_bits = precision.get_prec_bits()
    old_x64 = precision.jax_x64_enabled()
    try:
        precision.set_dps(80)
        assert precision.get_dps() == 80
        assert precision.get_prec_bits() == precision.dps_to_bits(80)

        precision.set_prec_bits(200)
        assert precision.get_prec_bits() == 200
        assert precision.get_dps() >= 60
        bits_before_workdps = precision.get_prec_bits()

        with precision.workdps(33):
            assert precision.get_dps() == 33
        assert precision.get_prec_bits() == precision.dps_to_bits(precision.get_dps())
        assert precision.get_prec_bits() != bits_before_workdps

        bits_before_workprec = precision.get_prec_bits()
        with precision.workprec(120):
            assert precision.get_prec_bits() == 120
        assert precision.get_prec_bits() == bits_before_workprec

        with precision.jax_x64_context(not old_x64):
            assert precision.jax_x64_enabled() is (not old_x64)
        assert precision.jax_x64_enabled() is old_x64
    finally:
        precision.set_dps(old_dps)
        precision.set_prec_bits(old_bits)
        precision.set_jax_x64(old_x64)


def test_eps_from_dps_monotonically_decreases_with_precision() -> None:
    eps_20 = precision.eps_from_dps(20)
    eps_50 = precision.eps_from_dps(50)
    eps_default = precision.eps_from_dps()

    assert eps_20.dtype == jnp.float64
    assert eps_50.dtype == jnp.float64
    assert eps_50 < eps_20
    assert eps_default > 0
