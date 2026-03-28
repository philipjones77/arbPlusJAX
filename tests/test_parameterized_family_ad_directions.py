from __future__ import annotations

import jax
import jax.numpy as jnp

from arbplusjax import api
from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import srb_mat


def test_core_scalar_pow_supports_value_and_parameter_ad() -> None:
    x = jnp.float32(1.3)
    y = jnp.float32(0.7)

    dx = jax.grad(lambda xv: api.eval_point("arb_pow", xv, y, dtype="float32"))(x)
    dy = jax.grad(lambda yv: api.eval_point("arb_pow", x, yv, dtype="float32"))(y)

    assert jnp.isfinite(dx)
    assert jnp.isfinite(dy)


def test_complex_hurwitz_zeta_supports_value_and_parameter_ad() -> None:
    s = jnp.complex128(2.2 + 0.1j)
    a = jnp.complex128(0.6 + 0.15j)

    ds = jax.grad(lambda sv: jnp.real(api.eval_point("acb_hurwitz_zeta", sv, a)))(s)
    da = jax.grad(lambda av: jnp.real(api.eval_point("acb_hurwitz_zeta", s, av)))(a)

    assert jnp.isfinite(jnp.real(ds))
    assert jnp.isfinite(jnp.imag(ds))
    assert jnp.isfinite(jnp.real(da))
    assert jnp.isfinite(jnp.imag(da))


def test_dense_operator_surface_supports_vector_and_scale_ad() -> None:
    base = jnp.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64)
    vec_mid = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    vec = di.interval(vec_mid, vec_mid)

    def loss_vec(v):
        plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(base, base))
        out = jrb_mat.jrb_mat_operator_plan_apply(plan, di.interval(v, v))
        return jnp.sum(di.midpoint(out))

    def loss_scale(s):
        plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(s * base, s * base))
        out = jrb_mat.jrb_mat_operator_plan_apply(plan, vec)
        return jnp.sum(di.midpoint(out))

    dvec = jax.grad(loss_vec)(vec_mid)
    dscale = jax.grad(loss_scale)(jnp.float64(1.0))

    assert jnp.all(jnp.isfinite(dvec))
    assert jnp.isfinite(dscale)


def test_sparse_surface_supports_vector_and_scale_ad() -> None:
    base = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=jnp.float64)
    vec = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)

    def loss_vec(v):
        sparse = srb_mat.srb_mat_from_dense_bcoo(base)
        out = api.eval_point("srb_mat_matvec", sparse, v)
        return jnp.sum(out)

    def loss_scale(s):
        sparse = srb_mat.srb_mat_from_dense_bcoo(s * base)
        out = api.eval_point("srb_mat_matvec", sparse, vec)
        return jnp.sum(out)

    dvec = jax.grad(loss_vec)(vec)
    dscale = jax.grad(loss_scale)(jnp.float64(1.0))

    assert jnp.all(jnp.isfinite(dvec))
    assert jnp.isfinite(dscale)


def test_matrix_free_surface_supports_rhs_and_shift_ad() -> None:
    base_diag = jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)
    a_mid = jnp.diag(base_diag)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(a_mid, a_mid))
    rhs_mid = jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float64)
    rhs = di.interval(rhs_mid, rhs_mid)

    def loss_rhs(v):
        solved = jrb_mat.jrb_mat_solve_action_point_jit(plan, di.interval(v, v), symmetric=True)
        return jnp.sum(di.midpoint(solved))

    def loss_shift(s):
        solved = jrb_mat.jrb_mat_multi_shift_solve_point(plan, rhs, jnp.asarray([s], dtype=jnp.float64), symmetric=True)
        return jnp.sum(di.midpoint(solved))

    drhs = jax.grad(loss_rhs)(rhs_mid)
    dshift = jax.grad(loss_shift)(jnp.float64(0.2))

    assert jnp.all(jnp.isfinite(drhs))
    assert jnp.isfinite(dshift)
