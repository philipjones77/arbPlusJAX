import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import mat_wrappers

from tests._test_checks import _check


def _real_spd_interval(a: jax.Array) -> jax.Array:
    return di.interval(a, a)


def _complex_box(a: jax.Array) -> jax.Array:
    return acb_core.acb_box(di.interval(jnp.real(a), jnp.real(a)), di.interval(jnp.imag(a), jnp.imag(a)))


def test_arb_dense_spd_modes_and_auto_route():
    mid = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs_mid = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
    a = _real_spd_interval(mid)
    b = _real_spd_interval(rhs_mid)

    point = mat_wrappers.arb_mat_spd_solve_mode(a, b, impl="point", prec_bits=53)
    basic = mat_wrappers.arb_mat_spd_solve_mode(a, b, impl="basic", prec_bits=53)
    rig = mat_wrappers.arb_mat_spd_solve_mode(a, b, impl="rigorous", prec_bits=53)
    adapt = mat_wrappers.arb_mat_spd_solve_mode(a, b, impl="adaptive", prec_bits=53)
    generic = arb_mat.arb_mat_solve(a, b)
    cho = mat_wrappers.arb_mat_cho_mode(a, impl="basic", prec_bits=53)
    ldl_l, ldl_d = mat_wrappers.arb_mat_ldl_mode(a, impl="basic", prec_bits=53)
    plan = mat_wrappers.arb_mat_dense_spd_solve_plan_prepare_mode(a, impl="basic", prec_bits=53)
    plan_out = mat_wrappers.arb_mat_dense_spd_solve_plan_apply_mode(plan, b, impl="basic", prec_bits=53)
    padded_plan = mat_wrappers.arb_mat_dense_spd_solve_plan_prepare_batch_mode_padded(jnp.stack([a, a], axis=0), pad_to=4, impl="basic", prec_bits=53)
    padded_out = arb_mat.arb_mat_dense_spd_solve_plan_apply_batch_padded(padded_plan, jnp.stack([b, b], axis=0), pad_to=4)

    expected = jnp.linalg.solve(mid, rhs_mid)
    chol_mid = di.midpoint(cho)
    l_mid = di.midpoint(ldl_l)
    d_mid = di.midpoint(ldl_d)
    recon = l_mid @ (jnp.eye(2, dtype=mid.dtype) * d_mid[..., None, :]) @ jnp.swapaxes(l_mid, -2, -1)

    _check(bool(mat_wrappers.arb_mat_is_symmetric_mode(a, impl="basic", prec_bits=53)))
    _check(bool(mat_wrappers.arb_mat_is_spd_mode(a, impl="basic", prec_bits=53)))
    _check(bool(jnp.allclose(point, expected)))
    _check(bool(jnp.allclose(di.midpoint(basic), expected)))
    _check(bool(jnp.allclose(di.midpoint(generic), expected)))
    _check(bool(jnp.all(di.contains(rig, basic))))
    _check(bool(jnp.all(di.contains(adapt, basic))))
    _check(bool(jnp.allclose(chol_mid @ jnp.swapaxes(chol_mid, -2, -1), mid)))
    _check(bool(jnp.allclose(recon, mid)))
    _check(bool(jnp.allclose(di.midpoint(plan_out), expected)))
    _check(padded_plan.factor.shape == (4, 2, 2, 2))
    _check(padded_out.shape == (4, 2, 1, 2))


def test_acb_dense_hpd_modes_and_auto_route():
    mid = jnp.array([[4.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 5.0 + 0.0j]], dtype=jnp.complex128)
    rhs_mid = jnp.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=jnp.complex128)
    a = _complex_box(mid)
    b = _complex_box(rhs_mid)

    point = mat_wrappers.acb_mat_hpd_solve_mode(a, b, impl="point", prec_bits=53)
    basic = mat_wrappers.acb_mat_hpd_solve_mode(a, b, impl="basic", prec_bits=53)
    rig = mat_wrappers.acb_mat_hpd_solve_mode(a, b, impl="rigorous", prec_bits=53)
    adapt = mat_wrappers.acb_mat_hpd_solve_mode(a, b, impl="adaptive", prec_bits=53)
    generic = acb_mat.acb_mat_solve(a, b)
    cho = mat_wrappers.acb_mat_cho_mode(a, impl="basic", prec_bits=53)
    ldl_l, ldl_d = mat_wrappers.acb_mat_ldl_mode(a, impl="basic", prec_bits=53)
    plan = mat_wrappers.acb_mat_dense_hpd_solve_plan_prepare_mode(a, impl="basic", prec_bits=53)
    plan_out = mat_wrappers.acb_mat_dense_hpd_solve_plan_apply_mode(plan, b, impl="basic", prec_bits=53)
    padded_plan = mat_wrappers.acb_mat_dense_hpd_solve_plan_prepare_batch_mode_padded(jnp.stack([a, a], axis=0), pad_to=4, impl="basic", prec_bits=53)
    padded_out = acb_mat.acb_mat_dense_hpd_solve_plan_apply_batch_padded(padded_plan, jnp.stack([b, b], axis=0), pad_to=4)

    expected = jnp.linalg.solve(mid, rhs_mid)
    chol_mid = acb_core.acb_midpoint(cho)
    l_mid = acb_core.acb_midpoint(ldl_l)
    d_mid = acb_core.acb_midpoint(ldl_d)
    recon = l_mid @ (jnp.eye(2, dtype=mid.dtype) * d_mid[..., None, :]) @ jnp.conj(jnp.swapaxes(l_mid, -2, -1))

    _check(bool(mat_wrappers.acb_mat_is_hermitian_mode(a, impl="basic", prec_bits=53)))
    _check(bool(mat_wrappers.acb_mat_is_hpd_mode(a, impl="basic", prec_bits=53)))
    _check(bool(jnp.allclose(point, expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(basic), expected)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(generic), expected)))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(rig), acb_core.acb_real(basic)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(rig), acb_core.acb_imag(basic)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_real(adapt), acb_core.acb_real(basic)))))
    _check(bool(jnp.all(di.contains(acb_core.acb_imag(adapt), acb_core.acb_imag(basic)))))
    _check(bool(jnp.allclose(chol_mid @ jnp.conj(jnp.swapaxes(chol_mid, -2, -1)), mid)))
    _check(bool(jnp.allclose(recon, mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(plan_out), expected)))
    _check(padded_plan.factor.shape == (4, 2, 2, 4))
    _check(padded_out.shape == (4, 2, 1, 4))
