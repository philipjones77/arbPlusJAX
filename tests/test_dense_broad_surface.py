import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import mat_wrappers

from tests._test_checks import _check


def _real_interval(a):
    return di.interval(a, a)


def _complex_box(a):
    return acb_core.acb_box(di.interval(jnp.real(a), jnp.real(a)), di.interval(jnp.imag(a), jnp.imag(a)))


def test_arb_dense_broad_modes_and_batches():
    a_mid = jnp.array([[2.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)
    b_mid = jnp.array([[1.5, -2.0], [4.0, 0.5]], dtype=jnp.float64)
    tril_mid = jnp.array([[2.0, 0.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs_mid = jnp.array([[2.0], [7.0]], dtype=jnp.float64)
    spd_mid = jnp.array([[4.0, 1.0], [1.0, 5.0]], dtype=jnp.float64)

    a = _real_interval(a_mid)
    b = _real_interval(b_mid)
    tril = _real_interval(tril_mid)
    rhs = _real_interval(rhs_mid)
    spd = _real_interval(spd_mid)

    add_point = mat_wrappers.arb_mat_add_mode(a, b, impl="point", prec_bits=53)
    add_basic = mat_wrappers.arb_mat_add_mode(a, b, impl="basic", prec_bits=53)
    mul_basic = mat_wrappers.arb_mat_mul_entrywise_mode(a, b, impl="basic", prec_bits=53)
    eigvals = mat_wrappers.arb_mat_eigvalsh_mode(spd, impl="basic", prec_bits=53)
    eigvals_pad = mat_wrappers.arb_mat_eigvalsh_batch_mode_padded(jnp.stack([spd, spd], axis=0), pad_to=4, impl="basic", prec_bits=53)
    solve_tril = mat_wrappers.arb_mat_solve_tril_mode(tril, rhs, impl="point", prec_bits=53)
    solve_tril_pad = mat_wrappers.arb_mat_solve_tril_batch_mode_padded(
        jnp.stack([tril, tril], axis=0),
        jnp.stack([rhs, rhs], axis=0),
        pad_to=4,
        impl="basic",
        prec_bits=53,
    )
    solve_lu = mat_wrappers.arb_mat_solve_lu_mode(a, rhs, impl="basic", prec_bits=53)

    _check(bool(jnp.allclose(add_point, a_mid + b_mid)))
    _check(bool(jnp.allclose(di.midpoint(add_basic), a_mid + b_mid)))
    _check(bool(jnp.allclose(di.midpoint(mul_basic), a_mid * b_mid)))
    _check(bool(mat_wrappers.arb_mat_is_diag_mode(_real_interval(jnp.diag(jnp.array([1.0, 2.0]))), impl="basic", prec_bits=53)))
    _check(bool(mat_wrappers.arb_mat_is_tril_mode(tril, impl="basic", prec_bits=53)))
    _check(bool(mat_wrappers.arb_mat_is_triu_mode(_real_interval(jnp.array([[2.0, 1.0], [0.0, 3.0]])), impl="basic", prec_bits=53)))
    _check(bool(jnp.allclose(di.midpoint(eigvals), jnp.linalg.eigvalsh(spd_mid))))
    _check(eigvals_pad.shape == (4, 2, 2))
    _check(bool(jnp.allclose(solve_tril, jnp.linalg.solve(tril_mid, rhs_mid))))
    _check(solve_tril_pad.shape == (4, 2, 1, 2))
    _check(bool(jnp.allclose(di.midpoint(solve_lu), jnp.linalg.solve(a_mid, rhs_mid))))


def test_acb_dense_broad_modes_and_batches():
    a_mid = jnp.array([[2.0 + 0.0j, 1.0 + 1.0j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=jnp.complex128)
    b_mid = jnp.array([[1.0 - 0.5j, -2.0 + 0.25j], [0.5 + 1.5j, 0.5 - 0.5j]], dtype=jnp.complex128)
    tril_mid = jnp.array([[2.0 + 0.0j, 0.0 + 0.0j], [1.0 - 0.5j, 3.0 + 0.0j]], dtype=jnp.complex128)
    rhs_mid = jnp.array([[2.0 + 1.0j], [7.0 - 0.5j]], dtype=jnp.complex128)
    hpd_mid = jnp.array([[5.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 4.0 + 0.0j]], dtype=jnp.complex128)

    a = _complex_box(a_mid)
    b = _complex_box(b_mid)
    tril = _complex_box(tril_mid)
    rhs = _complex_box(rhs_mid)
    hpd = _complex_box(hpd_mid)

    add_point = mat_wrappers.acb_mat_add_mode(a, b, impl="point", prec_bits=53)
    conj_basic = mat_wrappers.acb_mat_conjugate_mode(a, impl="basic", prec_bits=53)
    eigvals = mat_wrappers.acb_mat_eigvalsh_mode(hpd, impl="basic", prec_bits=53)
    eigvals_pad = mat_wrappers.acb_mat_eigvalsh_batch_mode_padded(jnp.stack([hpd, hpd], axis=0), pad_to=4, impl="basic", prec_bits=53)
    solve_triu = mat_wrappers.acb_mat_solve_triu_mode(
        _complex_box(jnp.array([[2.0 + 0.0j, 1.0 - 0.5j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=jnp.complex128)),
        rhs,
        impl="point",
        prec_bits=53,
    )
    solve_lu = mat_wrappers.acb_mat_solve_lu_mode(a, rhs, impl="basic", prec_bits=53)

    _check(bool(jnp.allclose(add_point, a_mid + b_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(conj_basic), jnp.conj(a_mid))))
    _check(bool(mat_wrappers.acb_mat_is_real_mode(_complex_box(jnp.eye(2, dtype=jnp.complex128)), impl="basic", prec_bits=53)))
    _check(bool(mat_wrappers.acb_mat_is_diag_mode(_complex_box(jnp.diag(jnp.array([1.0 + 0.0j, 2.0 + 0.0j]))), impl="basic", prec_bits=53)))
    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(eigvals)), jnp.linalg.eigvalsh(hpd_mid))))
    _check(eigvals_pad.shape == (4, 2, 4))
    _check(bool(jnp.allclose(solve_triu, jnp.linalg.solve(jnp.array([[2.0 + 0.0j, 1.0 - 0.5j], [0.0 + 0.0j, 3.0 + 0.0j]], dtype=jnp.complex128), rhs_mid))))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(solve_lu), jnp.linalg.solve(a_mid, rhs_mid))))


def test_dense_direct_batch_aliases():
    a_mid = jnp.array([[3.0, 1.0], [1.0, 4.0]], dtype=jnp.float64)
    a = _real_interval(a_mid)
    batch = jnp.stack([a, a], axis=0)
    eigvals = arb_mat.arb_mat_eigvalsh_batch_padded(batch, pad_to=4)
    _check(eigvals.shape == (4, 2, 2))

    z_mid = jnp.array([[3.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 4.0 + 0.0j]], dtype=jnp.complex128)
    z = _complex_box(z_mid)
    zbatch = jnp.stack([z, z], axis=0)
    conj_batch = acb_mat.acb_mat_conjugate_batch_padded(zbatch, pad_to=4)
    _check(conj_batch.shape == (4, 2, 2, 4))


def test_dense_matrix_functions_and_constructors():
    a_mid = jnp.array([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    a = _real_interval(a_mid)
    charpoly = mat_wrappers.arb_mat_charpoly_mode(a, impl="basic", prec_bits=53)
    pow2 = mat_wrappers.arb_mat_pow_ui_mode(a, 2, impl="point", prec_bits=53)
    expa = mat_wrappers.arb_mat_exp_mode(a, impl="basic", prec_bits=53)
    vals, vecs = jnp.linalg.eigh(a_mid)
    expected_exp = (vecs * jnp.exp(vals)[None, :]) @ vecs.T

    _check(bool(jnp.allclose(di.midpoint(charpoly), jnp.array([1.0, -4.0, 3.0], dtype=jnp.float64), atol=1e-8)))
    _check(bool(jnp.allclose(pow2, a_mid @ a_mid)))
    _check(bool(jnp.allclose(di.midpoint(expa), expected_exp, atol=1e-8)))

    coeffs = _real_interval(jnp.array([1.0, -3.0, 2.0], dtype=jnp.float64))
    companion = arb_mat.arb_mat_companion(coeffs)
    hilbert = arb_mat.arb_mat_hilbert(3)
    pascal = arb_mat.arb_mat_pascal(3)
    stirling = arb_mat.arb_mat_stirling(5)

    _check(companion.shape == (2, 2, 2))
    _check(bool(jnp.allclose(di.midpoint(hilbert)[0, 1], 0.5)))
    _check(bool(jnp.allclose(di.midpoint(pascal), jnp.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 6.0]], dtype=jnp.float64))))
    _check(bool(jnp.allclose(di.midpoint(stirling)[4, :5], jnp.array([0.0, 1.0, 7.0, 6.0, 1.0], dtype=jnp.float64))))


def test_complex_dense_matrix_functions_and_constructors():
    a_mid = jnp.array([[2.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 3.0 + 0.0j]], dtype=jnp.complex128)
    a = _complex_box(a_mid)
    charpoly = mat_wrappers.acb_mat_charpoly_mode(a, impl="basic", prec_bits=53)
    pow2 = mat_wrappers.acb_mat_pow_ui_mode(a, 2, impl="point", prec_bits=53)
    expa = mat_wrappers.acb_mat_exp_mode(a, impl="basic", prec_bits=53)
    vals, vecs = jnp.linalg.eigh(a_mid)
    expected_exp = (vecs * jnp.exp(vals)[None, :]) @ jnp.conj(vecs.T)

    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(charpoly)), jnp.array([1.0, -5.0, 4.0], dtype=jnp.float64), atol=1e-8)))
    _check(bool(jnp.allclose(pow2, a_mid @ a_mid)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(expa), expected_exp, atol=1e-8)))

    coeffs = _complex_box(jnp.array([1.0 + 0.0j, -3.0 + 0.0j, 2.0 + 0.0j], dtype=jnp.complex128))
    companion = acb_mat.acb_mat_companion(coeffs)
    hilbert = acb_mat.acb_mat_hilbert(3)
    pascal = acb_mat.acb_mat_pascal(3)
    stirling = acb_mat.acb_mat_stirling(5)

    _check(companion.shape == (2, 2, 4))
    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(hilbert)[0, 1]), 0.5)))
    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(pascal)), jnp.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 6.0]], dtype=jnp.float64))))
    _check(bool(jnp.allclose(jnp.real(acb_core.acb_midpoint(stirling)[4, :5]), jnp.array([0.0, 1.0, 7.0, 6.0, 1.0], dtype=jnp.float64))))
