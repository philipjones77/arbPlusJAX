import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import scb_mat
from arbplusjax import srb_mat


def test_dense_contract_helpers_and_cached_rmatvec_surface():
    dense_r = arb_mat.arb_mat_identity(3)
    vec_r = di.interval(jnp.arange(1.0, 4.0), jnp.arange(1.0, 4.0))
    dense_c = acb_mat.acb_mat_identity(3)
    vec_c = acb_core.acb_box(vec_r, di.interval(jnp.zeros((3,)), jnp.zeros((3,))))

    assert arb_mat.arb_mat_as_matrix(dense_r).shape == (3, 3, 2)
    assert arb_mat.arb_mat_as_vector(vec_r).shape == (3, 2)
    assert acb_mat.acb_mat_as_matrix(dense_c).shape == (3, 3, 4)
    assert acb_mat.acb_mat_as_vector(vec_c).shape == (3, 4)

    r_cache_r = arb_mat.arb_mat_rmatvec_cached_prepare(dense_r)
    r_cache_c = acb_mat.acb_mat_rmatvec_cached_prepare(dense_c)
    assert arb_mat.arb_mat_rmatvec_cached_apply(r_cache_r, vec_r).shape == (3, 2)
    assert acb_mat.acb_mat_rmatvec_cached_apply(r_cache_c, vec_c).shape == (3, 4)


def test_sparse_conversion_and_cached_rmatvec_surface():
    dense_r = jnp.eye(3, dtype=jnp.float64)
    dense_c = jnp.eye(3, dtype=jnp.complex128)

    s_r = srb_mat.srb_mat_from_dense_bcoo(dense_r)
    s_c = scb_mat.scb_mat_from_dense_bcoo(dense_c)
    v_r = jnp.arange(1.0, 4.0)
    v_c = jnp.arange(1.0, 4.0) + 0.0j

    r_plan_r = srb_mat.srb_mat_rmatvec_cached_prepare(s_r)
    r_plan_c = scb_mat.scb_mat_rmatvec_cached_prepare(s_c)
    assert srb_mat.srb_mat_rmatvec_cached_apply(r_plan_r, v_r).shape == (3,)
    assert scb_mat.scb_mat_rmatvec_cached_apply(r_plan_c, v_c).shape == (3,)


def test_matrix_free_contract_helpers_and_sparse_plan_surface():
    dense_r = arb_mat.arb_mat_identity(3)
    dense_c = acb_mat.acb_mat_identity(3)
    vec_r = di.interval(jnp.arange(1.0, 4.0), jnp.arange(1.0, 4.0))
    vec_c = acb_core.acb_box(vec_r, di.interval(jnp.zeros((3,)), jnp.zeros((3,))))

    assert jrb_mat.jrb_mat_as_interval_matrix(dense_r).shape == (3, 3, 2)
    assert jrb_mat.jrb_mat_as_interval_vector(vec_r).shape == (3, 2)
    assert jcb_mat.jcb_mat_as_box_matrix(dense_c).shape == (3, 3, 4)
    assert jcb_mat.jcb_mat_as_box_vector(vec_c).shape == (3, 4)

    sparse_r = srb_mat.srb_mat_from_dense_bcoo(jnp.eye(3, dtype=jnp.float64))
    sparse_c = scb_mat.scb_mat_from_dense_bcoo(jnp.eye(3, dtype=jnp.complex128))
    plan_r = jrb_mat.jrb_mat_sparse_operator_rmatvec_plan_prepare(sparse_r)
    plan_c = jcb_mat.jcb_mat_sparse_operator_rmatvec_plan_prepare(sparse_c)
    assert jrb_mat.jrb_mat_operator_plan_apply(plan_r, vec_r).shape == (3, 2)
    assert jcb_mat.jcb_mat_operator_plan_apply(plan_c, vec_c).shape == (3, 4)
