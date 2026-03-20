import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di

from tests._test_checks import _check


def _ibox(a):
    return di.interval(a, a)


def _cbox(a):
    return acb_core.acb_box(di.interval(jnp.real(a), jnp.real(a)), di.interval(jnp.imag(a), jnp.imag(a)))


def test_arb_dense_aliases_and_symmetric_eigh():
    mid = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
    a = _ibox(mid)
    b = _ibox(rhs)

    tril = arb_mat.arb_mat_solve_tril(a, b)
    triu = arb_mat.arb_mat_solve_triu(_ibox(jnp.array([[4.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)), b)
    lu = arb_mat.arb_mat_solve_lu(a, b)
    vals = arb_mat.arb_mat_eigvalsh(a)
    vals_prec = arb_mat.arb_mat_eigvalsh_prec(a, prec_bits=53)
    vals2, vecs = arb_mat.arb_mat_eigh(a)

    ref_vals, ref_vecs = jnp.linalg.eigh(mid)

    _check(bool(jnp.allclose(di.midpoint(lu), jnp.linalg.solve(mid, rhs))))
    _check(tril.shape == (2, 1, 2))
    _check(triu.shape == (2, 1, 2))
    _check(vals.shape == (2, 2))
    _check(vecs.shape == (2, 2, 2))
    _check(bool(jnp.allclose(di.midpoint(vals), ref_vals)))
    _check(bool(jnp.allclose(di.midpoint(vals_prec), ref_vals)))
    _check(bool(jnp.allclose(di.midpoint(vals2), ref_vals)))
    _check(bool(jnp.allclose(jnp.abs(di.midpoint(vecs)), jnp.abs(ref_vecs))))


def test_acb_dense_aliases_and_hermitian_eigh():
    mid = jnp.array([[4.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 5.0 + 0.0j]], dtype=jnp.complex128)
    rhs = jnp.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=jnp.complex128)
    a = _cbox(mid)
    b = _cbox(rhs)

    lu = acb_mat.acb_mat_solve_lu(a, b)
    vals = acb_mat.acb_mat_eigvalsh(a)
    vals_prec = acb_mat.acb_mat_eigvalsh_prec(a, prec_bits=53)
    vals2, vecs = acb_mat.acb_mat_eigh(a)

    ref_vals, ref_vecs = jnp.linalg.eigh(mid)

    _check(bool(jnp.allclose(acb_core.acb_midpoint(lu), jnp.linalg.solve(mid, rhs))))
    _check(vals.shape == (2, 4))
    _check(vecs.shape == (2, 2, 4))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals), ref_vals)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals_prec), ref_vals)))
    _check(bool(jnp.allclose(acb_core.acb_midpoint(vals2), ref_vals)))
    _check(bool(jnp.allclose(jnp.abs(acb_core.acb_midpoint(vecs)), jnp.abs(ref_vecs))))
