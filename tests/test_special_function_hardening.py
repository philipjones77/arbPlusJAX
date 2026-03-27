from __future__ import annotations

import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import api
from arbplusjax import barnesg
from arbplusjax import double_interval as di
from arbplusjax import hypgeom
from arbplusjax import hypgeom_wrappers
from arbplusjax import stable_kernels
from arbplusjax.special.gamma.barnes_double_gamma_ifj import log_barnesdoublegamma_ifj


def _box(re_lo: float, re_hi: float, im_lo: float, im_hi: float):
    return acb_core.acb_box(
        di.interval(jnp.float64(re_lo), jnp.float64(re_hi)),
        di.interval(jnp.float64(im_lo), jnp.float64(im_hi)),
    )


def _box_exact(x: jnp.ndarray):
    return acb_core.acb_box(
        di.interval(jnp.real(x), jnp.real(x)),
        di.interval(jnp.imag(x), jnp.imag(x)),
    )


def test_incomplete_bessel_i_high_precision_refine_batch_matches_scalar_fragile_regime():
    nu = jnp.asarray([12.0, 13.0], dtype=jnp.float64)
    z = jnp.asarray([9.0, 9.5], dtype=jnp.float64)
    upper = jnp.asarray([jnp.pi - 0.05, jnp.pi - 0.04], dtype=jnp.float64)

    batch = api.incomplete_bessel_i_batch(nu, z, upper, mode="point", method="high_precision_refine")
    scalar = jnp.asarray(
        [
            api.incomplete_bessel_i(nu[0], z[0], upper[0], mode="point", method="high_precision_refine"),
            api.incomplete_bessel_i(nu[1], z[1], upper[1], mode="point", method="high_precision_refine"),
        ]
    )

    assert jnp.all(jnp.isfinite(batch))
    assert jnp.allclose(batch, scalar, rtol=1e-10, atol=1e-10)


def test_incomplete_bessel_i_auto_batch_falls_back_to_quadrature_under_vmap():
    nu = jnp.asarray([12.0, 13.0], dtype=jnp.float64)
    z = jnp.asarray([9.0, 9.5], dtype=jnp.float64)
    upper = jnp.asarray([jnp.pi - 0.05, jnp.pi - 0.04], dtype=jnp.float64)

    auto = api.incomplete_bessel_i_batch(nu, z, upper, mode="point", method="auto")
    quadrature = api.incomplete_bessel_i_batch(nu, z, upper, mode="point", method="quadrature")

    assert jnp.all(jnp.isfinite(auto))
    assert jnp.allclose(auto, quadrature, rtol=1e-10, atol=1e-10)


def test_ifj_log_barnes_double_gamma_respects_shift_recurrence():
    z = jnp.asarray(0.7 + 0.1j, dtype=jnp.complex128)
    tau = jnp.asarray(1.0, dtype=jnp.float64)

    lhs = log_barnesdoublegamma_ifj(z + tau, tau, dps=60) - log_barnesdoublegamma_ifj(z, tau, dps=60)
    rhs = barnesg._complex_loggamma(z)

    assert jnp.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_barnes_provider_alias_tracks_ifj_and_diagnostics_in_hardened_regime():
    z = jnp.asarray(0.7 + 0.1j, dtype=jnp.complex128)
    tau = jnp.asarray(1.0, dtype=jnp.float64)

    provider = stable_kernels.barnesdoublegamma(z, tau, dps=60)
    ifj = api.eval_point("ifj_barnesdoublegamma", z, tau, dps=60)
    diagnostics = api.eval_point("ifj_barnesdoublegamma_diagnostics", z, tau, dps=60, max_m_cap=96)

    assert jnp.allclose(provider, ifj, rtol=1e-10, atol=1e-10)
    assert diagnostics.m_used <= diagnostics.max_m_cap
    assert diagnostics.n_shift >= 1


def test_hypgeom_complex_incomplete_gamma_regularized_complement_consistency():
    s = _box(1.2, 1.25, -0.05, 0.05)
    z = _box(0.3, 0.35, -0.02, 0.02)
    lower = hypgeom_wrappers.acb_hypgeom_gamma_lower_mode(s, z, impl="rigorous", prec_bits=53, regularized=True)
    upper = hypgeom_wrappers.acb_hypgeom_gamma_upper_mode(s, z, impl="adaptive", prec_bits=53, regularized=True)
    total = acb_core.acb_add(lower, upper)

    real_ok = bool(acb_core.acb_real(total)[0] <= 1.0 <= acb_core.acb_real(total)[1])
    imag_ok = bool(acb_core.acb_imag(total)[0] <= 0.0 <= acb_core.acb_imag(total)[1])
    assert real_ok
    assert imag_ok


def test_hypgeom_u_hardened_point_and_mode_surfaces_stay_aligned_while_pfq_mode_stays_finite():
    u_a = jnp.asarray([1.1, 1.2, 1.3], dtype=jnp.float64)
    u_b = jnp.asarray([2.1, 2.2, 2.3], dtype=jnp.float64)
    u_z = jnp.asarray([0.6, 0.8, 1.0], dtype=jnp.float64)
    u_point = api.bind_point_batch_jit("hypgeom.arb_hypgeom_u", dtype="float64", pad_to=8)(u_a, u_b, u_z)
    u_mode = hypgeom_wrappers.arb_hypgeom_u_batch_mode_padded(
        di.interval(u_a, u_a),
        di.interval(u_b, u_b),
        di.interval(u_z, u_z),
        pad_to=8,
        impl="adaptive",
        prec_bits=53,
    )

    pfq_a = jnp.asarray([[0.6, 0.9], [0.7, 1.0], [0.8, 1.1]], dtype=jnp.float64)
    pfq_b = jnp.asarray([[1.4], [1.5], [1.6]], dtype=jnp.float64)
    pfq_z = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)
    pfq_point = api.bind_point_batch_jit("hypgeom.arb_hypgeom_pfq", dtype="float64", pad_to=8)(pfq_a, pfq_b, pfq_z)
    pfq_mode = hypgeom_wrappers.arb_hypgeom_pfq_batch_mode_padded(
        di.interval(pfq_a, pfq_a),
        di.interval(pfq_b, pfq_b),
        di.interval(pfq_z, pfq_z),
        pad_to=8,
        impl="adaptive",
        prec_bits=53,
    )

    assert jnp.allclose(di.midpoint(u_mode[: u_point.shape[0]]), u_point, rtol=1e-8, atol=1e-8)
    assert pfq_mode.shape[0] >= pfq_point.shape[0]
    assert bool(jnp.all(jnp.isfinite(di.midpoint(pfq_mode[: pfq_point.shape[0]]))))


def test_hypgeom_pfq_point_surface_matches_family_owned_real_and_complex_implementations():
    pfq_a = jnp.asarray([[0.6, 0.9], [0.7, 1.0], [0.8, 1.1]], dtype=jnp.float64)
    pfq_b = jnp.asarray([[1.4], [1.5], [1.6]], dtype=jnp.float64)
    pfq_z = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float64)

    point_real = api.bind_point_batch_jit("hypgeom.arb_hypgeom_pfq", dtype="float64", pad_to=8)(pfq_a, pfq_b, pfq_z)
    family_real = jnp.asarray(
        [
            di.midpoint(
                hypgeom.arb_hypgeom_pfq(
                    di.interval(a_row, a_row),
                    di.interval(b_row, b_row),
                    di.interval(z_val, z_val),
                )
            )
            for a_row, b_row, z_val in zip(pfq_a, pfq_b, pfq_z, strict=True)
        ],
        dtype=jnp.float64,
    )
    assert jnp.allclose(point_real, family_real, rtol=1e-10, atol=1e-10)

    pfq_ac = pfq_a.astype(jnp.complex128)
    pfq_bc = pfq_b.astype(jnp.complex128)
    pfq_zc = pfq_z.astype(jnp.complex128)
    point_complex = api.bind_point_batch_jit("hypgeom.acb_hypgeom_pfq", pad_to=8)(pfq_ac, pfq_bc, pfq_zc)
    family_complex = jnp.asarray(
        [
            acb_core.acb_midpoint(hypgeom.acb_hypgeom_pfq(_box_exact(a_row), _box_exact(b_row), _box_exact(z_val)))
            for a_row, b_row, z_val in zip(pfq_ac, pfq_bc, pfq_zc, strict=True)
        ],
        dtype=jnp.complex128,
    )
    assert jnp.allclose(point_complex, family_complex, rtol=1e-10, atol=1e-10)
