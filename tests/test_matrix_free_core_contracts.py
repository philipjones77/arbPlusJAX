import jax.numpy as jnp

from arbplusjax import double_interval as di
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
    oriented_shell = mfc.oriented_shell_preconditioner_plan(
        context={
            "forward_callback": lambda v, ctx: ctx["diag"] * v,
            "transpose_callback": lambda v, ctx: ctx["diag_t"] * v,
            "diag": jnp.asarray([2.0, 3.0], dtype=jnp.float64),
            "diag_t": jnp.asarray([5.0, 7.0], dtype=jnp.float64),
        },
        algebra="jrb",
        orientation="forward",
        forward_callback=lambda v, ctx: ctx["diag"] * v,
        transpose_callback=lambda v, ctx: ctx["diag_t"] * v,
    )

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
    assert jnp.allclose(
        mfc.preconditioner_plan_apply(oriented_shell, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        jnp.asarray([2.0, -3.0], dtype=jnp.float64),
    )
    oriented_shell_t = mfc.preconditioner_transpose_plan(oriented_shell)
    assert oriented_shell_t is not None
    assert jnp.allclose(
        mfc.preconditioner_plan_apply(oriented_shell_t, vec, midpoint_vector=_midpoint_vector, sparse_bcoo_matvec=_sparse_bcoo_matvec, dtype=jnp.float64),
        jnp.asarray([5.0, -7.0], dtype=jnp.float64),
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
    converged_count, locked_count, deflated_count, converged = mfc.eig_convergence_summary(
        jnp.asarray([1e-12, 2e-3], dtype=jnp.float64),
        tol=1e-6,
        requested=1,
    )
    assert int(converged_count) == 1
    assert int(locked_count) == 1
    assert int(deflated_count) == 1
    assert bool(converged)

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


def test_contour_rational_and_logdet_solve_helpers_have_stable_contracts():
    x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    contour = mfc.contour_integral_action_point(
        lambda shift, v: v / (shift + 1.0),
        x,
        center=1.0 + 0.0j,
        radius=0.5,
        quadrature_order=6,
        node_weight_fn=lambda node: node,
    )
    assert contour.shape == x.shape
    assert jnp.all(jnp.isfinite(jnp.real(contour)))

    poly = mfc.polynomial_spectral_action_midpoint(
        lambda v: 2.0 * v,
        x,
        jnp.asarray([1.0, 0.5], dtype=jnp.float64),
        coeff_dtype=jnp.float64,
    )
    assert jnp.allclose(poly, x + 0.5 * (2.0 * x))

    rational = mfc.rational_spectral_action_midpoint(
        lambda v: 2.0 * v,
        lambda shift, v: v / (shift + 2.0),
        x,
        shifts=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        weights=jnp.asarray([1.0, -0.5], dtype=jnp.float64),
        polynomial_coefficients=jnp.asarray([1.0], dtype=jnp.float64),
        coeff_dtype=jnp.float64,
    )
    expected = x + (1.0 / 2.0) * x - 0.5 * ((1.0 / 3.0) * x)
    assert jnp.allclose(rational, expected)

    result = mfc.combine_logdet_solve_point(
        operator="op",
        transpose_operator="op_t",
        rhs=x,
        probes=jnp.stack([x, -x], axis=0),
        solve_with_diagnostics=lambda operator, rhs_value: (rhs_value / 2.0, {"operator": operator, "kind": "solve"}),
        logdet_with_diagnostics=lambda operator, probe_value: (jnp.asarray(3.0), {"operator": operator, "kind": "logdet", "count": probe_value.shape[0]}),
        solver="cg",
        implicit_adjoint=True,
        structured="symmetric",
        algebra="jrb",
    )
    assert jnp.allclose(result.solve, x / 2.0)
    assert jnp.allclose(result.logdet, jnp.asarray(3.0))
    assert result.aux.operator == "op"
    assert result.aux.transpose_operator == "op_t"
    assert result.aux.solve_diagnostics["kind"] == "solve"
    assert result.aux.logdet_diagnostics["count"] == 2
    assert result.aux.implicit_adjoint


def test_slq_and_hutchpp_metadata_helpers_have_stable_contracts():
    projected = jnp.stack(
        [
            jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64)),
            jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64)),
        ],
        axis=0,
    )
    beta0 = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    samples = jnp.asarray([jnp.log(2.0) + jnp.log(3.0), jnp.log(2.0) + jnp.log(3.0)], dtype=jnp.float64)
    slq = mfc.make_slq_quadrature_metadata(projected, beta0, samples, hermitian=True, target_stderr=1e-3, min_probes=2, max_probes=8, block_size=2)
    assert slq.nodes.shape == (2, 2)
    assert slq.weights.shape == (2, 2)
    assert bool(jnp.allclose(slq.statistics.mean, samples[0]))
    assert int(slq.statistics.probe_count) == 2
    assert int(slq.statistics.recommended_probe_count) % 2 == 0

    heat = mfc.slq_heat_trace(slq.nodes[0], slq.weights[0], 0.5)
    hist = mfc.slq_spectral_density(slq.nodes[0], slq.weights[0], jnp.asarray([1.0, 2.5, 4.0], dtype=jnp.float64), normalize=True)
    assert bool(jnp.isfinite(heat))
    assert hist.shape == (2,)
    assert bool(jnp.allclose(jnp.sum(hist), 1.0, atol=1e-6))

    hutch = mfc.make_hutchpp_trace_metadata(
        basis=jnp.eye(2, dtype=jnp.float64),
        low_rank_trace=jnp.asarray(4.0, dtype=jnp.float64),
        residual_samples=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
        target_stderr=1e-3,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    assert hutch.basis.shape == (2, 2)
    assert bool(jnp.allclose(hutch.low_rank_trace + hutch.residual_trace, 5.0))
    assert int(hutch.statistics.recommended_probe_count) % 2 == 0
    assert bool(mfc.probe_statistics_target_met(hutch.statistics, target_stderr=1.0))
    assert bool(mfc.probe_statistics_should_stop(hutch.statistics, target_stderr=1.0, max_probes=8))


def test_probe_statistics_and_correction_expansion_helpers_have_stable_contracts():
    stats = mfc.make_probe_estimate_statistics(
        jnp.asarray([2.0, 2.0, 2.0, 2.0], dtype=jnp.float64),
        target_stderr=1e-3,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    assert bool(mfc.probe_statistics_target_met(stats, target_stderr=1e-3))
    assert bool(mfc.probe_statistics_should_stop(stats, target_stderr=1e-3, max_probes=8))

    basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float64)
    vecs = basis
    residuals = jnp.asarray([[0.0], [2.0]], dtype=jnp.float64)
    expanded = mfc.expand_subspace_with_corrections(
        basis,
        vecs,
        residuals,
        orthonormalize_columns_fn=lambda arr: jnp.linalg.qr(arr, mode="reduced")[0],
        target_cols=2,
    )
    assert expanded.shape == (2, 2)
    assert bool(jnp.allclose(expanded.T @ expanded, jnp.eye(2, dtype=jnp.float64), atol=1e-6))


def test_deflated_operator_metadata_and_trace_helpers_have_stable_contracts():
    action_fn = lambda probe: probe * jnp.asarray([4.0, 9.0], dtype=jnp.float64)
    sketch = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    residual = jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)

    deflation = mfc.prepare_deflated_operator_metadata_point(
        action_fn,
        sketch,
        coerce_probes=jnp.asarray,
        midpoint_value=jnp.asarray,
        point_from_midpoint=jnp.asarray,
        basis_dtype=jnp.float64,
        trace_inner=lambda q, fq_cols: jnp.trace(q.T @ fq_cols),
    )
    assert deflation.basis.shape == (2, 1)
    assert bool(jnp.allclose(deflation.low_rank_trace, 4.0, atol=1e-6))

    residual_apply = mfc.deflated_operator_apply_midpoint(
        jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        deflation=deflation,
        apply_operator_midpoint=lambda v: action_fn(v),
    )
    assert bool(jnp.allclose(residual_apply, jnp.asarray([0.0, 4.0], dtype=jnp.float64), atol=1e-6))

    metadata = mfc.deflated_trace_estimate_from_metadata_point(
        action_fn,
        deflation,
        residual,
        coerce_probes=jnp.asarray,
        midpoint_value=jnp.asarray,
        point_from_midpoint=jnp.asarray,
        residual_project=lambda z, q: z - (z @ q) @ q.T,
        quadratic_reduce=lambda z_proj, hz: jnp.sum(z_proj * hz, axis=-1),
    )
    assert bool(jnp.allclose(mfc.hutchpp_trace_from_metadata(metadata), 13.0, atol=1e-6))


def test_generic_slq_and_hutchpp_consolidation_helpers_have_stable_contracts():
    def action_fn(v):
        return di.interval(di.midpoint(v), di.midpoint(v))

    probes = di.interval(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
    )
    hutch = mfc.hutchpp_trace_with_metadata_projected_point(
        action_fn,
        probes,
        probes,
        coerce_probes=di.as_interval,
        midpoint_value=di.midpoint,
        point_from_midpoint=lambda x: di.interval(x, x),
        basis_dtype=jnp.float64,
        trace_inner=lambda q, fq: jnp.trace(q.T @ fq),
        residual_project=lambda z, q: z - (z @ q) @ q.T,
        quadratic_reduce=lambda z, hz: jnp.sum(z * hz, axis=-1),
        zero_scalar=jnp.asarray(0.0, dtype=jnp.float64),
        target_stderr=1e-3,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    assert hutch.basis.shape[-1] <= 2
    assert int(hutch.statistics.recommended_probe_count) % 2 == 0

    def lanczos_stub(v, steps):
        del v
        diag = jnp.diag(jnp.linspace(2.0, 3.0, steps, dtype=jnp.float64))
        return jnp.zeros((steps, steps), dtype=jnp.float64), diag, jnp.asarray(1.0, dtype=jnp.float64)

    slq = mfc.slq_prepare_metadata_point(
        lanczos_stub,
        probes,
        2,
        coerce_probes=di.as_interval,
        hermitian=True,
        scalar_fun=jnp.log,
        target_stderr=1e-3,
        min_probes=2,
        max_probes=8,
        block_size=2,
    )
    assert slq.nodes.shape == (2, 2)
    assert bool(jnp.isfinite(mfc.slq_heat_trace_from_metadata(slq, 0.5)))


def test_implicit_krylov_solve_midpoint_uses_custom_linear_solve_contract():
    dense = jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float64)
    operator = mfc.dense_operator_plan(dense, orientation="forward", algebra="jrb")
    rhs = jnp.asarray([2.0, -8.0], dtype=jnp.float64)

    x_mid, info, residual, rhs_norm, meta = mfc.implicit_krylov_solve_midpoint(
        operator,
        rhs,
        solver="cg",
        structured="symmetric",
        midpoint_vector=_midpoint_vector,
        lift_vector=_identity_point_from_midpoint,
        sparse_bcoo_matvec=_sparse_bcoo_matvec,
        dtype=jnp.float64,
    )

    assert jnp.allclose(x_mid, jnp.asarray([1.0, -2.0], dtype=jnp.float64), atol=1e-6)
    assert bool(info["converged"])
    assert float(residual) <= 1e-6
    assert float(rhs_norm) > 0.0
    assert meta.implicit_adjoint
    assert meta.transpose_operator is not None


def test_implicit_krylov_solve_midpoint_uses_transpose_shell_preconditioner_contract():
    dense = jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float64)
    operator = mfc.dense_operator_plan(dense, orientation="forward", algebra="jrb")
    rhs = jnp.asarray([2.0, -8.0], dtype=jnp.float64)
    preconditioner = mfc.oriented_shell_preconditioner_plan(
        context={
            "forward_callback": lambda v, ctx: ctx["diag"] * v,
            "transpose_callback": lambda v, ctx: ctx["diag_t"] * v,
            "diag": jnp.asarray([0.5, 0.25], dtype=jnp.float64),
            "diag_t": jnp.asarray([0.5, 0.25], dtype=jnp.float64),
        },
        algebra="jrb",
        orientation="forward",
        forward_callback=lambda v, ctx: ctx["diag"] * v,
        transpose_callback=lambda v, ctx: ctx["diag_t"] * v,
    )

    x_mid, info, residual, rhs_norm, meta = mfc.implicit_krylov_solve_midpoint(
        operator,
        rhs,
        solver="cg",
        structured="symmetric",
        midpoint_vector=_midpoint_vector,
        lift_vector=_identity_point_from_midpoint,
        sparse_bcoo_matvec=_sparse_bcoo_matvec,
        dtype=jnp.float64,
        preconditioner=preconditioner,
    )

    assert jnp.allclose(x_mid, jnp.asarray([1.0, -2.0], dtype=jnp.float64), atol=1e-6)
    assert bool(info["converged"])
    assert float(residual) <= 1e-6
    assert float(rhs_norm) > 0.0
    assert meta.transpose_preconditioner is not None


def test_orthogonal_probe_block_helpers_return_orthonormal_midpoints():
    key = jnp.asarray([0, 321], dtype=jnp.uint32)

    real_block = mfc.orthogonal_rademacher_probe_block_real(_identity_point_from_midpoint, 4, key=key, num=2)
    real_mid = jnp.asarray(real_block)
    assert real_mid.shape == (2, 4)
    assert jnp.allclose(real_mid @ real_mid.T, jnp.eye(2, dtype=jnp.float64), atol=1e-6)

    complex_block = mfc.orthogonal_normal_probe_block_complex(_identity_point_from_midpoint, 4, key=key, num=2)
    complex_mid = jnp.asarray(complex_block)
    assert complex_mid.shape == (2, 4)
    assert jnp.allclose(complex_mid @ jnp.conjugate(complex_mid.T), jnp.eye(2, dtype=jnp.complex128), atol=1e-6)


def test_probe_statistics_and_adaptive_budget_helpers_have_stable_contracts():
    real_samples = jnp.asarray([1.0, 3.0, 5.0], dtype=jnp.float64)
    mean, variance, stderr = mfc.probe_sample_statistics(real_samples)
    assert jnp.allclose(mean, jnp.asarray(3.0))
    assert jnp.allclose(variance, jnp.asarray(4.0))
    assert stderr > 0.0

    complex_samples = jnp.asarray([1.0 + 1.0j, 3.0 + 1.0j], dtype=jnp.complex128)
    mean_c, variance_c, stderr_c = mfc.probe_sample_statistics(complex_samples)
    assert jnp.allclose(mean_c, jnp.asarray(2.0 + 1.0j))
    assert variance_c > 0.0
    assert stderr_c > 0.0

    recommended = mfc.adaptive_probe_count_from_pilot(
        real_samples,
        target_stderr=0.5,
        min_probes=3,
        max_probes=16,
        block_size=2,
    )
    assert int(recommended) >= 3
    assert int(recommended) <= 16
    assert int(recommended) % 2 == 0
