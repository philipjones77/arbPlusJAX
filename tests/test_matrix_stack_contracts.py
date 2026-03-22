import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import matrix_free_core
from arbplusjax import scb_mat
from arbplusjax import scb_block_mat
from arbplusjax import scb_vblock_mat
from arbplusjax import srb_mat
from arbplusjax import srb_block_mat
from arbplusjax import srb_vblock_mat


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
    assert arb_mat.arb_mat_operator_plan_prepare(dense_r).algebra == "jrb"
    assert acb_mat.acb_mat_operator_plan_prepare(dense_c).algebra == "jcb"


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
    assert srb_mat.srb_mat_operator_plan_prepare(s_r).algebra == "jrb"
    assert scb_mat.scb_mat_operator_plan_prepare(s_c).algebra == "jcb"


def test_block_and_vblock_storage_modules_adapt_into_operator_plans():
    dense_r = jnp.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 10.0],
        ],
        dtype=jnp.float64,
    )
    vec_r = di.interval(jnp.asarray([1.0, -1.0, 0.5, 2.0]), jnp.asarray([1.0, -1.0, 0.5, 2.0]))
    block_r = srb_block_mat.srb_block_mat_from_dense_csr(dense_r, block_shape=(2, 2))
    vblock_r = srb_vblock_mat.srb_vblock_mat_from_dense_csr(
        dense_r,
        row_block_sizes=jnp.asarray([1, 1, 2], dtype=jnp.int32),
        col_block_sizes=jnp.asarray([1, 1, 2], dtype=jnp.int32),
    )
    block_plan_r = srb_block_mat.srb_block_mat_operator_plan_prepare(block_r)
    vblock_plan_r = srb_vblock_mat.srb_vblock_mat_operator_plan_prepare(vblock_r)
    assert jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(block_plan_r, vec_r)), dense_r @ di.midpoint(vec_r), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(di.midpoint(jrb_mat.jrb_mat_operator_plan_apply(vblock_plan_r, vec_r)), dense_r @ di.midpoint(vec_r), rtol=1e-6, atol=1e-6)

    dense_c = dense_r.astype(jnp.complex128) + 1j * jnp.eye(4, dtype=jnp.complex128)
    vec_c_mid = jnp.asarray([1.0 + 0.5j, -1.0 + 0.0j, 0.5 - 0.5j, 2.0 + 0.0j], dtype=jnp.complex128)
    vec_c = acb_core.acb_box(di.interval(jnp.real(vec_c_mid), jnp.real(vec_c_mid)), di.interval(jnp.imag(vec_c_mid), jnp.imag(vec_c_mid)))
    block_c = scb_block_mat.scb_block_mat_from_dense_csr(dense_c, block_shape=(2, 2))
    vblock_c = scb_vblock_mat.scb_vblock_mat_from_dense_csr(
        dense_c,
        row_block_sizes=jnp.asarray([1, 1, 2], dtype=jnp.int32),
        col_block_sizes=jnp.asarray([1, 1, 2], dtype=jnp.int32),
    )
    block_plan_c = scb_block_mat.scb_block_mat_operator_plan_prepare(block_c)
    vblock_plan_c = scb_vblock_mat.scb_vblock_mat_operator_plan_prepare(vblock_c)
    assert jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(block_plan_c, vec_c)), dense_c @ vec_c_mid, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(acb_core.acb_midpoint(jcb_mat.jcb_mat_operator_plan_apply(vblock_plan_c, vec_c)), dense_c @ vec_c_mid, rtol=1e-6, atol=1e-6)


def test_matrix_free_contract_helpers_and_sparse_plan_surface():
    dense_r = arb_mat.arb_mat_identity(3)
    dense_c = acb_mat.acb_mat_identity(3)
    vec_r = di.interval(jnp.arange(1.0, 4.0), jnp.arange(1.0, 4.0))
    vec_c = acb_core.acb_box(vec_r, di.interval(jnp.zeros((3,)), jnp.zeros((3,))))

    assert jrb_mat.jrb_mat_as_interval_matrix(dense_r).shape == (3, 3, 2)
    assert jrb_mat.jrb_mat_as_interval_vector(vec_r).shape == (3, 2)
    assert jcb_mat.jcb_mat_as_box_matrix(dense_c).shape == (3, 3, 4)
    assert jcb_mat.jcb_mat_as_box_vector(vec_c).shape == (3, 4)

    sparse_r = jrb_mat.sparse_common.dense_to_sparse_bcoo(jnp.eye(3, dtype=jnp.float64), algebra="jrb")
    sparse_c = jcb_mat.sparse_common.dense_to_sparse_bcoo(jnp.eye(3, dtype=jnp.complex128), algebra="jcb")
    plan_r = jrb_mat.jrb_mat_sparse_operator_rmatvec_plan_prepare(sparse_r)
    plan_c = jcb_mat.jcb_mat_sparse_operator_rmatvec_plan_prepare(sparse_c)
    assert jrb_mat.jrb_mat_operator_plan_apply(plan_r, vec_r).shape == (3, 2)
    assert jcb_mat.jcb_mat_operator_plan_apply(plan_c, vec_c).shape == (3, 4)


def test_matrix_free_operator_owned_eigsh_surface_from_sparse_plans():
    dense_r = jnp.diag(jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float64))
    dense_c = jnp.diag(jnp.asarray([1.5, 3.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)

    sparse_r = jrb_mat.sparse_common.dense_to_sparse_bcoo(dense_r, algebra="jrb")
    sparse_c = jcb_mat.sparse_common.dense_to_sparse_bcoo(dense_c, algebra="jcb")
    plan_r = jrb_mat.jrb_mat_sparse_operator_plan_prepare(sparse_r)
    plan_c = jcb_mat.jcb_mat_sparse_operator_plan_prepare(sparse_c)

    vals_r, vecs_r = jrb_mat.jrb_mat_eigsh_point(plan_r, size=3, k=2, which="smallest", steps=3)
    vals_c, vecs_c = jcb_mat.jcb_mat_eigsh_point(plan_c, size=3, k=2, which="largest", steps=3)

    assert jnp.allclose(vals_r, jnp.asarray([1.0, 2.0], dtype=jnp.float64), rtol=1e-8, atol=1e-8)
    assert jnp.allclose(vals_c, jnp.asarray([3.0, 5.0], dtype=jnp.float64), rtol=1e-8, atol=1e-8)
    assert vecs_r.shape == (3, 2)
    assert vecs_c.shape == (3, 2)


def test_matrix_free_core_shifted_and_recycled_plan_pytrees():
    dense_r = arb_mat.arb_mat_identity(3)
    op = jrb_mat.jrb_mat_dense_operator_plan_prepare(dense_r)
    prec = matrix_free_core.identity_preconditioner_plan(size=3, dtype=jnp.float64, algebra="jrb")
    shifted = matrix_free_core.make_shifted_solve_plan(
        op,
        jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        preconditioner=prec,
        solver="cg",
        algebra="jrb",
        structured="spd",
    )
    recycled = matrix_free_core.make_recycled_krylov_state(
        basis=jnp.eye(3, dtype=jnp.float64),
        projected=jnp.eye(3, dtype=jnp.float64),
        residual=jnp.zeros((3,), dtype=jnp.float64),
        preconditioner=prec,
        algorithm="lanczos",
        algebra="jrb",
        structured="spd",
    )

    shifted_leaves, shifted_treedef = jax.tree_util.tree_flatten(shifted)
    rebuilt_shifted = jax.tree_util.tree_unflatten(shifted_treedef, shifted_leaves)
    assert rebuilt_shifted.solver == "cg"
    assert rebuilt_shifted.algebra == "jrb"
    assert rebuilt_shifted.structured == "spd"
    assert jnp.allclose(rebuilt_shifted.shifts, jnp.asarray([0.1, 0.2], dtype=jnp.float64))

    recycled_leaves, recycled_treedef = jax.tree_util.tree_flatten(recycled)
    rebuilt_recycled = jax.tree_util.tree_unflatten(recycled_treedef, recycled_leaves)
    assert rebuilt_recycled.algorithm == "lanczos"
    assert rebuilt_recycled.algebra == "jrb"
    assert rebuilt_recycled.structured == "spd"
    assert rebuilt_recycled.basis.shape == (3, 3)


def test_matrix_free_core_multi_shift_solve_identity_case():
    dense_r = arb_mat.arb_mat_identity(3)
    op = jrb_mat.jrb_mat_dense_operator_plan_prepare(dense_r)
    shifted = matrix_free_core.make_shifted_solve_plan(
        op,
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        preconditioner=matrix_free_core.identity_preconditioner_plan(size=3, dtype=jnp.float64, algebra="jrb"),
        solver="cg",
        algebra="jrb",
        structured="spd",
    )
    rhs = di.interval(jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([1.0, 2.0, 3.0]))
    out = matrix_free_core.multi_shift_solve_point(
        shifted,
        rhs,
        apply_operator=jrb_mat.iterative_solvers,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
        tol=1e-10,
    )
    assert out.shape == (2, 3)
    assert jnp.allclose(out[0], jnp.asarray([1.0, 2.0, 3.0]), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(out[1], jnp.asarray([0.5, 1.0, 1.5]), rtol=1e-6, atol=1e-6)


def test_matrix_free_core_jacobi_preconditioner_plans_apply_diagonal_inverse():
    dense = jnp.diag(jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float64))
    dense_prec = matrix_free_core.dense_jacobi_preconditioner_plan(dense, algebra="jrb")
    vec = di.interval(jnp.asarray([2.0, 8.0, 16.0]), jnp.asarray([2.0, 8.0, 16.0]))
    dense_out = matrix_free_core.preconditioner_plan_apply(
        dense_prec,
        vec,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    assert jnp.allclose(dense_out, jnp.asarray([1.0, 2.0, 2.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)

    sparse = jrb_mat.sparse_common.dense_to_sparse_bcoo(dense, algebra="jrb")
    sparse_prec = matrix_free_core.sparse_bcoo_jacobi_preconditioner_plan(
        sparse,
        as_sparse_bcoo=jrb_mat.sparse_common.as_sparse_bcoo,
        algebra="jrb",
    )
    sparse_out = matrix_free_core.preconditioner_plan_apply(
        sparse_prec,
        vec,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    assert jnp.allclose(sparse_out, dense_out, rtol=1e-6, atol=1e-6)


def test_matrix_free_shell_and_finite_difference_plans_live_in_operator_layer():
    vec_r = di.interval(jnp.asarray([1.0, -2.0, 3.0]), jnp.asarray([1.0, -2.0, 3.0]))
    shell_r = jrb_mat.jrb_mat_shell_operator_plan_prepare(lambda v: 2.0 * v)
    shell_prec_r = jrb_mat.jrb_mat_shell_preconditioner_plan_prepare(lambda v: 0.5 * v)
    out_r = jrb_mat.jrb_mat_operator_plan_apply(shell_r, vec_r)
    prec_r = matrix_free_core.preconditioner_plan_apply(
        shell_prec_r,
        vec_r,
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    assert jnp.allclose(di.midpoint(out_r), 2.0 * di.midpoint(vec_r), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(prec_r, 0.5 * di.midpoint(vec_r), rtol=1e-6, atol=1e-6)

    base_r = jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float64)
    fd_r = jrb_mat.jrb_mat_finite_difference_operator_plan_prepare(lambda u: u * u, base_point=base_r)
    jvp_r = jrb_mat.jrb_mat_operator_plan_apply(fd_r, vec_r)
    assert jnp.allclose(di.midpoint(jvp_r), 2.0 * base_r * di.midpoint(vec_r), rtol=2e-3, atol=2e-3)
    fd_r2 = jrb_mat.jrb_mat_finite_difference_operator_plan_set_base(fd_r, base_point=jnp.asarray([2.0, 0.0, 1.0], dtype=jnp.float64))
    jvp_r2 = jrb_mat.jrb_mat_operator_plan_apply(fd_r2, vec_r)
    assert jnp.allclose(di.midpoint(jvp_r2), 2.0 * jnp.asarray([2.0, 0.0, 1.0]) * di.midpoint(vec_r), rtol=2e-3, atol=2e-3)
    fd_prec_r = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(fd_r)
    fd_prec_out_r = matrix_free_core.preconditioner_plan_apply(
        fd_prec_r,
        di.interval(jnp.asarray([2.0, 8.0, 2.0]), jnp.asarray([2.0, 8.0, 2.0])),
        midpoint_vector=di.midpoint,
        sparse_bcoo_matvec=jrb_mat.sparse_common.sparse_bcoo_matvec,
        dtype=jnp.float64,
    )
    assert jnp.allclose(fd_prec_out_r, jnp.asarray([1.0, 2.0, -1.0], dtype=jnp.float64), rtol=2e-3, atol=2e-3)

    vec_c = acb_core.acb_box(vec_r, di.interval(jnp.zeros((3,)), jnp.zeros((3,))))
    shell_c = jcb_mat.jcb_mat_shell_operator_plan_prepare(lambda v: (1.0 + 1.0j) * v)
    out_c = jcb_mat.jcb_mat_operator_plan_apply(shell_c, vec_c)
    assert jnp.allclose(acb_core.acb_midpoint(out_c), (1.0 + 1.0j) * acb_core.acb_midpoint(vec_c), rtol=1e-6, atol=1e-6)

    base_c = jnp.asarray([1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.5j], dtype=jnp.complex128)
    fd_c = jcb_mat.jcb_mat_finite_difference_operator_plan_prepare(lambda u: u * u, base_point=base_c)
    jvp_c = jcb_mat.jcb_mat_operator_plan_apply(fd_c, vec_c)
    assert jnp.allclose(acb_core.acb_midpoint(jvp_c), 2.0 * base_c * acb_core.acb_midpoint(vec_c), rtol=2e-3, atol=2e-3)


def test_matrix_free_parametric_operator_plans_are_parameter_differentiable():
    indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    x_r = di.interval(jnp.asarray([1.5, -2.0]), jnp.asarray([1.5, -2.0]))
    data_r = jnp.asarray([2.0, -1.0, 3.0, 4.0], dtype=jnp.float64)

    def objective_r(data):
        plan = jrb_mat.jrb_mat_bcoo_parametric_operator_plan_prepare(indices, data, shape=(2, 2))
        y = jrb_mat.jrb_mat_operator_plan_apply(plan, x_r)
        return jnp.sum(di.midpoint(y))

    grad_r = jax.grad(objective_r)(data_r)
    assert jnp.allclose(grad_r, jnp.asarray([1.5, -2.0, 1.5, -2.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)

    x_c = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, -2.0]), jnp.asarray([1.0, -2.0])),
        di.interval(jnp.asarray([0.5, 1.0]), jnp.asarray([0.5, 1.0])),
    )
    data_c = jnp.asarray([2.0 + 0.5j, -1.0 + 1.0j, 3.0 - 2.0j, 4.0 + 0.0j], dtype=jnp.complex128)

    def objective_c(data):
        plan = jcb_mat.jcb_mat_bcoo_parametric_operator_plan_prepare(indices, data, shape=(2, 2))
        y = jcb_mat.jcb_mat_operator_plan_apply(plan, x_c)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    grad_c = jax.grad(objective_c)(data_c)
    expected_c = acb_core.acb_midpoint(x_c)[jnp.asarray([0, 1, 0, 1])]
    assert jnp.allclose(grad_c, expected_c, rtol=1e-6, atol=1e-6)


def test_matrix_free_dense_parametric_operator_plans_are_parameter_differentiable():
    dense_r = di.interval(
        jnp.asarray([[2.0, -1.0], [3.0, 4.0]], dtype=jnp.float64),
        jnp.asarray([[2.0, -1.0], [3.0, 4.0]], dtype=jnp.float64),
    )
    x_r = di.interval(jnp.asarray([1.5, -2.0]), jnp.asarray([1.5, -2.0]))

    def objective_r(mid):
        plan = jrb_mat.jrb_mat_dense_parametric_operator_plan_prepare(di.interval(mid, mid))
        y = jrb_mat.jrb_mat_operator_plan_apply(plan, x_r)
        return jnp.sum(di.midpoint(y))

    grad_r = jax.grad(objective_r)(di.midpoint(dense_r))
    expected_r = jnp.broadcast_to(di.midpoint(x_r)[None, :], grad_r.shape)
    assert jnp.allclose(grad_r, expected_r, rtol=1e-6, atol=1e-6)

    dense_c_mid = jnp.asarray([[2.0 + 0.5j, -1.0 + 1.0j], [3.0 - 2.0j, 4.0 + 0.0j]], dtype=jnp.complex128)
    x_c = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, -2.0]), jnp.asarray([1.0, -2.0])),
        di.interval(jnp.asarray([0.5, 1.0]), jnp.asarray([0.5, 1.0])),
    )

    def objective_c(mid):
        boxed = acb_core.acb_box(di.interval(jnp.real(mid), jnp.real(mid)), di.interval(jnp.imag(mid), jnp.imag(mid)))
        plan = jcb_mat.jcb_mat_dense_parametric_operator_plan_prepare(boxed)
        y = jcb_mat.jcb_mat_operator_plan_apply(plan, x_c)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    grad_c = jax.grad(objective_c)(dense_c_mid)
    expected_c = jnp.broadcast_to(acb_core.acb_midpoint(x_c)[None, :], grad_c.shape)
    assert jnp.allclose(grad_c, expected_c, rtol=1e-6, atol=1e-6)


def test_matrix_free_shell_operator_plan_context_is_parameter_differentiable():
    x_r = di.interval(jnp.asarray([2.0, -1.0]), jnp.asarray([2.0, -1.0]))

    def objective_r(scale):
        plan = jrb_mat.jrb_mat_shell_operator_plan_prepare(
            lambda v, ctx: jnp.asarray(ctx["scale"], dtype=jnp.float64) * v,
            context={"scale": scale},
        )
        y = jrb_mat.jrb_mat_operator_plan_apply(plan, x_r)
        return jnp.sum(di.midpoint(y))

    grad_r = jax.grad(objective_r)(jnp.asarray(3.0, dtype=jnp.float64))
    assert jnp.allclose(grad_r, jnp.asarray(1.0, dtype=jnp.float64), rtol=1e-6, atol=1e-6)

    x_c = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, -2.0]), jnp.asarray([1.0, -2.0])),
        di.interval(jnp.asarray([0.5, 1.0]), jnp.asarray([0.5, 1.0])),
    )

    def objective_c(scale):
        plan = jcb_mat.jcb_mat_shell_operator_plan_prepare(
            lambda v, ctx: jnp.asarray(ctx["scale"], dtype=jnp.complex128) * v,
            context={"scale": scale},
        )
        y = jcb_mat.jcb_mat_operator_plan_apply(plan, x_c)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    grad_c = jax.grad(objective_c)(jnp.asarray(1.0 + 2.0j, dtype=jnp.complex128))
    assert jnp.allclose(grad_c, jnp.asarray(-1.0 + 1.5j, dtype=jnp.complex128), rtol=1e-6, atol=1e-6)


def test_sparse_interval_and_box_storage_public_wrappers_round_trip():
    dense_r = jnp.asarray([[2.0, 0.0], [-1.0, 3.0]], dtype=jnp.float64)
    sparse_r = srb_mat.srb_mat_from_dense_csr(dense_r)
    iv_sparse = srb_mat.srb_mat_to_interval_sparse(sparse_r)
    dense_iv = srb_mat.srb_mat_interval_to_dense(iv_sparse)
    vec_iv = di.interval(jnp.asarray([1.0, 2.0]), jnp.asarray([1.0, 2.0]))
    y_iv = srb_mat.srb_mat_interval_matvec(iv_sparse, vec_iv)
    assert jnp.allclose(di.midpoint(dense_iv), dense_r, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(di.midpoint(y_iv), dense_r @ di.midpoint(vec_iv), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(di.midpoint(srb_mat.srb_mat_interval_to_dense(srb_mat.srb_mat_interval_transpose(iv_sparse))), dense_r.T, rtol=1e-6, atol=1e-6)

    dense_c = jnp.asarray([[2.0 + 0.5j, 0.0], [-1.0j, 3.0 - 0.25j]], dtype=jnp.complex128)
    sparse_c = scb_mat.scb_mat_from_dense_csr(dense_c)
    box_sparse = scb_mat.scb_mat_to_box_sparse(sparse_c)
    dense_box = scb_mat.scb_mat_box_to_dense(box_sparse)
    vec_box = acb_core.acb_box(
        di.interval(jnp.asarray([1.0, 2.0]), jnp.asarray([1.0, 2.0])),
        di.interval(jnp.asarray([0.5, -1.0]), jnp.asarray([0.5, -1.0])),
    )
    y_box = scb_mat.scb_mat_box_matvec(box_sparse, vec_box)
    assert jnp.allclose(acb_core.acb_midpoint(dense_box), dense_c, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(acb_core.acb_midpoint(y_box), dense_c @ acb_core.acb_midpoint(vec_box), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(acb_core.acb_midpoint(scb_mat.scb_mat_box_to_dense(scb_mat.scb_mat_box_conjugate_transpose(box_sparse))), jnp.conjugate(dense_c.T), rtol=1e-6, atol=1e-6)


def test_matrix_free_logdet_solve_surfaces_match_diagonal_identity_cases():
    dense_r = di.interval(jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64)), jnp.diag(jnp.asarray([2.0, 4.0], dtype=jnp.float64)))
    rhs_r = di.interval(jnp.asarray([2.0, 8.0], dtype=jnp.float64), jnp.asarray([2.0, 8.0], dtype=jnp.float64))
    plan_r = jrb_mat.jrb_mat_dense_operator_plan_prepare(dense_r)
    probes_r = jnp.stack(
        [
            rhs_r,
            di.interval(jnp.asarray([1.0, 1.0], dtype=jnp.float64), jnp.asarray([1.0, 1.0], dtype=jnp.float64)),
        ],
        axis=0,
    )
    result_r = jrb_mat.jrb_mat_logdet_solve_point(plan_r, rhs_r, probes_r, steps=2, symmetric=True)
    direct_logdet_r = jrb_mat.jrb_mat_logdet_slq_point(plan_r, probes_r, 2)
    direct_solve_r = jrb_mat.jrb_mat_solve_action_point(plan_r, rhs_r, symmetric=True)
    assert jnp.allclose(result_r.logdet, direct_logdet_r, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(di.midpoint(result_r.solve), di.midpoint(direct_solve_r), rtol=1e-6, atol=1e-6)
    assert bool(result_r.aux.solve_diagnostics.converged)

    dense_c_mid = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64)).astype(jnp.complex128)
    dense_c = acb_core.acb_box(di.interval(jnp.real(dense_c_mid), jnp.real(dense_c_mid)), di.interval(jnp.imag(dense_c_mid), jnp.imag(dense_c_mid)))
    rhs_c_mid = jnp.asarray([2.0 + 0.0j, 15.0 + 0.0j], dtype=jnp.complex128)
    rhs_c = acb_core.acb_box(di.interval(jnp.real(rhs_c_mid), jnp.real(rhs_c_mid)), di.interval(jnp.imag(rhs_c_mid), jnp.imag(rhs_c_mid)))
    plan_c = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense_c)
    probes_c = jnp.stack(
        [
            rhs_c,
            acb_core.acb_box(
                di.interval(jnp.asarray([1.0, 1.0], dtype=jnp.float64), jnp.asarray([1.0, 1.0], dtype=jnp.float64)),
                di.interval(jnp.asarray([0.0, 0.0], dtype=jnp.float64), jnp.asarray([0.0, 0.0], dtype=jnp.float64)),
            ),
        ],
        axis=0,
    )
    result_c = jcb_mat.jcb_mat_logdet_solve_point(plan_c, rhs_c, probes_c, steps=2, hermitian=True)
    direct_logdet_c = jcb_mat.jcb_mat_logdet_slq_point(plan_c, probes_c, 2)
    direct_solve_c = jcb_mat.jcb_mat_solve_action_point(plan_c, rhs_c, hermitian=True)
    assert jnp.allclose(result_c.logdet, direct_logdet_c, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(acb_core.acb_midpoint(result_c.solve), acb_core.acb_midpoint(direct_solve_c), rtol=1e-6, atol=1e-6)
    assert bool(result_c.aux.solve_diagnostics.converged)
