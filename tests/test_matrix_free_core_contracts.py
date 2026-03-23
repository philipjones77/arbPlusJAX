import jax.numpy as jnp

from arbplusjax import matrix_free_core as mfc


def _midpoint_vector(x):
    return jnp.asarray(x)


def _identity_point_from_midpoint(x):
    return jnp.asarray(x)


def _sparse_bcoo_matvec(plan, v, *, algebra, label):
    del algebra, label
    rows = jnp.asarray(plan.indices[:, 0], dtype=jnp.int32)
    cols = jnp.asarray(plan.indices[:, 1], dtype=jnp.int32)
    vals = jnp.asarray(plan.data) * jnp.asarray(v)[cols]
    return jnp.zeros((plan.rows,), dtype=jnp.asarray(v).dtype).at[rows].add(vals)


def test_operator_and_preconditioner_plans_cover_dense_shell_and_scaled_paths():
    dense = jnp.asarray([[2.0, 1.0], [0.0, 3.0]], dtype=jnp.float64)
    vec = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    forward = mfc.dense_operator_plan(dense, orientation="forward", algebra="jrb")
    transpose = mfc.dense_operator_plan(dense, orientation="transpose", algebra="jrb")
    shell = mfc.shell_operator_plan(lambda v, ctx: ctx["scale"] * v, context={"scale": 2.0}, algebra="jrb")
    scaled = mfc.scaled_operator(forward, 0.5)

    assert jnp.allclose(
        mfc.operator_plan_apply(forward, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        dense @ vec,
    )
    assert jnp.allclose(
        mfc.operator_plan_apply(transpose, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        dense.T @ vec,
    )
    assert jnp.allclose(
        mfc.operator_plan_apply(shell, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        2.0 * vec,
    )
    assert jnp.allclose(
        mfc.operator_apply_midpoint(scaled, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        0.5 * (dense @ vec),
    )

    identity = mfc.identity_preconditioner_plan(size=2, dtype=jnp.float64, algebra="jrb")
    diagonal = mfc.diagonal_preconditioner_plan(jnp.asarray([2.0, 0.5], dtype=jnp.float64), algebra="jrb")
    dense_prec = mfc.dense_preconditioner_plan(jnp.asarray([[1.0, 0.0], [0.0, 4.0]], dtype=jnp.float64), algebra="jrb")

    assert jnp.allclose(
        mfc.preconditioner_plan_apply(identity, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        vec,
    )
    assert jnp.allclose(
        mfc.preconditioner_plan_apply(diagonal, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        jnp.asarray([2.0, 0.5], dtype=jnp.float64) * vec,
    )
    assert jnp.allclose(
        mfc.preconditioner_plan_apply(dense_prec, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        jnp.asarray([1.0, -4.0], dtype=jnp.float64),
    )


def test_finite_difference_operator_and_jacobi_helpers_track_base_updates():
    matrix = jnp.asarray([[3.0, 0.0], [0.0, 5.0]], dtype=jnp.float64)
    base_point = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    direction = jnp.asarray([0.25, -0.5], dtype=jnp.float64)

    def linear_fn(x):
        return matrix @ x

    plan = mfc.finite_difference_operator_plan(linear_fn, base_point=base_point, algebra="jrb", relative_error=1e-6, umin=1e-6)
    applied = mfc.operator_plan_apply(plan, direction, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64)
    assert jnp.allclose(applied, matrix @ direction, rtol=1e-4, atol=1e-6)

    updated = mfc.finite_difference_operator_plan_set_base(plan, base_point=2.0 * base_point, base_value=linear_fn(2.0 * base_point))
    applied_updated = mfc.operator_plan_apply(updated, direction, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64)
    assert jnp.allclose(applied_updated, matrix @ direction, rtol=1e-4, atol=1e-6)

    jacobi = mfc.finite_difference_jacobi_preconditioner_plan(
        updated,
        midpoint_vector=_midpoint_vector,
        sparse_bcoo_matvec=_sparse_bcoo_matvec,
        dtype=jnp.float64,
        algebra="jrb",
    )
    assert jnp.allclose(jacobi.payload, jnp.asarray([1.0 / 3.0, 1.0 / 5.0], dtype=jnp.float64), rtol=1e-4, atol=1e-6)


def test_restart_eigensolver_and_action_helpers_have_stable_contracts():
    basis = jnp.eye(2, dtype=jnp.float64)
    diag = jnp.asarray([4.0, 1.0], dtype=jnp.float64)

    def apply_block(q):
        return diag[:, None] * q

    restarted = mfc.restarted_subspace_iteration_point(
        apply_block,
        basis,
        subspace_iters=1,
        restarts=2,
        k=1,
        which="largest",
        hermitian=True,
    )
    gram = restarted.T @ restarted
    assert restarted.shape == basis.shape
    assert jnp.allclose(gram, jnp.eye(2, dtype=jnp.float64), atol=1e-6)

    residuals = jnp.asarray([[1e-12, 1e-3], [0.0, 2e-3]], dtype=jnp.float64)
    evals = jnp.asarray([5.0, 1.0], dtype=jnp.float64)
    order = mfc.eig_restart_column_order(evals, residuals, which="largest", lock_tol=1e-8)
    assert tuple(sorted(jnp.asarray(order).tolist())) == (0, 1)
    assert mfc.eig_restart_lock_tolerance(steps=4, restarts=2) >= 1e-10

    action = mfc.restarted_action_point(lambda x: 2.0 * x, jnp.asarray([1.0, -1.0], dtype=jnp.float64), restarts=3)
    assert jnp.allclose(action, jnp.asarray([8.0, -8.0], dtype=jnp.float64))


def test_contour_probe_and_point_helpers_preserve_shape_contracts():
    nodes, weights = mfc.contour_quadrature_nodes(1.0 + 0.0j, 2.0, quadrature_order=8)
    assert nodes.shape == (8,)
    assert weights.shape == (8,)

    filtered = mfc.contour_filter_subspace_point(
        lambda shift, q: q / (shift + 1.0),
        jnp.eye(2, dtype=jnp.complex128),
        center=1.0 + 0.0j,
        radius=0.5,
        quadrature_order=6,
    )
    assert filtered.shape == (2, 2)

    real_apply = lambda x: 3.0 * x
    complex_vec = jnp.asarray([1.0 + 2.0j, -0.5 + 0.25j], dtype=jnp.complex128)
    assert jnp.allclose(mfc.complexify_real_linear_operator(real_apply, complex_vec), 3.0 * complex_vec)

    applied_point = mfc.operator_apply_point(
        lambda x: 2.0 * x,
        jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        midpoint_apply=lambda op, x: op(x),
        coerce_vector=jnp.asarray,
        point_from_midpoint=_identity_point_from_midpoint,
        full_like=jnp.zeros_like,
        finite_mask_fn=jnp.isfinite,
        dtype=jnp.float64,
    )
    assert jnp.allclose(applied_point, jnp.asarray([2.0, -4.0], dtype=jnp.float64))

    key = jnp.asarray([0, 123], dtype=jnp.uint32)
    probes_real = mfc.rademacher_probes_real(_identity_point_from_midpoint, 4, key=key, num=3)
    probes_complex = mfc.normal_probes_complex(_identity_point_from_midpoint, 4, key=key, num=2)
    assert probes_real.shape == (3, 4)
    assert probes_complex.shape == (2, 4)
