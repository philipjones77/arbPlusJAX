import jax
import jax.numpy as jnp

from arbplusjax import curvature
from arbplusjax import double_interval as di
from arbplusjax import jrb_mat
from arbplusjax import jcb_mat
from arbplusjax import sparse_common


def _mat2(a00, a01, a10, a11):
    return jnp.asarray([[a00, a01], [a10, a11]])


def _imat2(a00, a01, a10, a11):
    row0 = jnp.stack([di.interval(a00, a00), di.interval(a01, a01)], axis=0)
    row1 = jnp.stack([di.interval(a10, a10), di.interval(a11, a11)], axis=0)
    return jnp.stack([row0, row1], axis=0)


def _ivec2(x0, x1):
    xs = jnp.asarray([x0, x1], dtype=jnp.float64)
    return di.interval(xs, xs)


def _cbox(z):
    z = jnp.asarray(z, dtype=jnp.complex128)
    return jnp.stack([jnp.real(z), jnp.real(z), jnp.imag(z), jnp.imag(z)], axis=-1)


def _cvec2(z0, z1):
    return jnp.stack([_cbox(z0), _cbox(z1)], axis=0)


def test_dense_curvature_operator_supports_dense_actions_and_diagnostics():
    dense = _mat2(2.0, 0.5, 0.5, 3.0)
    op = curvature.make_dense_curvature_operator(dense, psd=True)
    x = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    assert jnp.allclose(op.matvec(x), dense @ x)
    assert jnp.allclose(op.solve(x), jnp.linalg.solve(dense, x))
    assert jnp.allclose(op.diagonal(), jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    assert jnp.allclose(op.logdet(), jnp.linalg.slogdet(dense)[1])
    assert bool(curvature.dot_test_curvature(op, x, x))
    assert bool(op.is_psd())
    assert float(curvature.estimate_condition_number(op)) >= 1.0


def test_hvp_hessian_and_posterior_precision_builders_share_operator_contract():
    fun = lambda z: 0.5 * jnp.dot(z, jnp.asarray([2.0, 3.0], dtype=jnp.float64) * z)
    x = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    v = jnp.asarray([0.25, -0.5], dtype=jnp.float64)

    hv = curvature.hvp(fun, x, v)
    assert jnp.allclose(hv, jnp.asarray([0.5, -1.5], dtype=jnp.float64))

    hess = curvature.make_hessian_operator(fun, x, dense=True)
    assert jnp.allclose(hess.matvec(v), hv)

    prior = curvature.make_dense_curvature_operator(jnp.diag(jnp.asarray([1.0, 1.0], dtype=jnp.float64)), psd=True)
    post = curvature.make_posterior_precision_operator(prior, hess, damping=0.5)
    assert jnp.allclose(post.matvec(v), jnp.asarray([3.5 * 0.25, 4.5 * -0.5], dtype=jnp.float64))
    assert bool(curvature.ensure_psd(post).is_psd())
    assert jnp.allclose(post.solve(v), jnp.asarray([0.25 / 3.5, -0.5 / 4.5], dtype=jnp.float64))
    assert jnp.allclose(post.diagonal(), jnp.asarray([3.5, 4.5], dtype=jnp.float64))
    assert jnp.allclose(post.logdet(), jnp.log(3.5) + jnp.log(4.5))


def test_dense_hvp_operator_exposes_exact_phase1_actions():
    fun = lambda z: 0.5 * jnp.dot(z, jnp.asarray([2.0, 3.0], dtype=jnp.float64) * z)
    x = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    rhs = jnp.asarray([0.5, -1.5], dtype=jnp.float64)

    op = curvature.make_hvp_operator(fun, x, dense=True, psd=True)
    dense = jnp.diag(jnp.asarray([2.0, 3.0], dtype=jnp.float64))

    assert jnp.allclose(op.to_dense(), dense)
    assert jnp.allclose(op.diagonal(), jnp.asarray([2.0, 3.0], dtype=jnp.float64))
    assert jnp.allclose(op.trace(), 5.0)
    assert jnp.allclose(op.solve(rhs), jnp.linalg.solve(dense, rhs))
    assert jnp.allclose(op.logdet(), jnp.log(2.0) + jnp.log(3.0))
    assert jnp.allclose(op.inverse_diagonal(), jnp.asarray([0.5, 1.0 / 3.0], dtype=jnp.float64))
    assert bool(op.is_psd())


def test_curvature_spec_flows_into_operator_and_posterior_precision_metadata():
    spec = curvature.CurvatureSpec(
        kind="posterior_precision",
        representation="operator",
        differentiation_mode="forward_over_reverse",
        damping=0.25,
        jitter=0.5,
        symmetrize=True,
        enforce_psd=True,
    )
    dense = _mat2(2.0, 0.0, 0.0, 3.0)
    prior = curvature.make_dense_curvature_operator(dense, psd=True)
    like = curvature.make_dense_curvature_operator(jnp.eye(2, dtype=jnp.float64), psd=True)
    post = curvature.make_posterior_precision_operator(prior, like, damping=0.25, jitter=0.5, spec=spec)

    assert post.metadata["kind"] == "posterior_precision"
    assert post.metadata["representation"] == "operator"
    assert post.metadata["differentiation_mode"] == "forward_over_reverse"
    assert post.metadata["damping"] == 0.25
    assert post.metadata["jitter"] == 0.5
    assert bool(post.is_symmetric())
    assert bool(post.is_psd())


def test_curvature_composition_helpers_preserve_dense_backed_operator_surface():
    dense = _mat2(2.0, 0.25, 0.75, 3.0)
    op = curvature.make_dense_curvature_operator(dense, symmetric=False, psd=None)
    vec = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    sym = curvature.symmetrize_operator(op)
    sym_dense = 0.5 * (dense + dense.T)
    assert jnp.allclose(sym.matvec(vec), sym_dense @ vec)
    assert jnp.allclose(sym.solve(vec), jnp.linalg.solve(sym_dense, vec))
    assert jnp.allclose(sym.diagonal(), jnp.diag(sym_dense))
    assert jnp.allclose(sym.logdet(), jnp.linalg.slogdet(sym_dense)[1])
    assert jnp.allclose(sym.inverse_diagonal(), jnp.diag(jnp.linalg.inv(sym_dense)))

    jittered = curvature.add_jitter(sym, 0.5)
    jittered_dense = sym_dense + 0.5 * jnp.eye(2, dtype=jnp.float64)
    assert jnp.allclose(jittered.solve(vec), jnp.linalg.solve(jittered_dense, vec))
    assert jnp.allclose(jittered.diagonal(), jnp.diag(jittered_dense))
    assert jnp.allclose(jittered.logdet(), jnp.linalg.slogdet(jittered_dense)[1])

    psd = curvature.ensure_psd(op, jitter=0.5)
    assert bool(psd.is_psd())
    assert bool(psd.is_symmetric())
    assert jnp.allclose(psd.solve(vec), jnp.linalg.solve(jittered_dense, vec))
    grad = jnp.asarray([1.0, -3.0], dtype=jnp.float64)
    assert jnp.allclose(curvature.newton_step(grad, sym, damping=0.5), jnp.linalg.solve(jittered_dense, -grad))


def test_jrb_curvature_operator_reuses_matrix_free_solve_and_logdet_surfaces():
    dense = _imat2(2.0, 0.0, 0.0, 3.0)
    plan = jrb_mat.jrb_mat_dense_operator(dense)
    probes = jnp.stack([_ivec2(1.0, 1.0), _ivec2(1.0, -1.0)], axis=0)
    op = curvature.make_jrb_curvature_operator(plan, shape=(2, 2), probes=probes, steps=2, symmetric=True)
    rhs = jnp.asarray([2.0, -3.0], dtype=jnp.float64)

    assert jnp.allclose(op.solve(rhs), jnp.asarray([1.0, -1.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(op.logdet(), jnp.log(2.0) + jnp.log(3.0), rtol=1e-6, atol=1e-6)


def test_jcb_curvature_operator_reuses_hermitian_matrix_free_surfaces():
    dense = jnp.stack([_cvec2(2.0 + 0.0j, 0.0 + 0.0j), _cvec2(0.0 + 0.0j, 3.0 + 0.0j)], axis=0)
    plan = jcb_mat.jcb_mat_dense_operator(dense)
    probes = jnp.stack([_cvec2(1.0 + 0.0j, 1.0 + 0.0j), _cvec2(1.0 + 0.0j, -1.0 + 0.0j)], axis=0)
    op = curvature.make_jcb_curvature_operator(plan, shape=(2, 2), probes=probes, steps=2, hermitian=True)
    rhs = jnp.asarray([2.0 + 0.0j, -3.0 + 0.0j], dtype=jnp.complex128)

    assert jnp.allclose(op.solve(rhs), jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j], dtype=jnp.complex128), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(op.logdet(), jnp.log(2.0) + jnp.log(3.0), rtol=1e-6, atol=1e-6)


def test_selected_inverse_delegates_to_sparse_real_path():
    data = jnp.asarray([2.0, 1.0, 1.0, 3.0], dtype=jnp.float64)
    indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    bcoo = sparse_common.SparseBCOO(data=data, indices=indices, rows=2, cols=2, algebra="jrb")
    dense = jnp.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    op = curvature.make_curvature_operator(
        shape=(2, 2),
        dtype=jnp.float64,
        matvec=lambda v: dense @ v,
        rmatvec=lambda v: dense.T @ v,
        to_dense_fn=lambda: dense,
        metadata={"symmetric": True, "psd": True, "sparse_bcoo": bcoo},
        inverse_diagonal_fn=lambda **kwargs: jnp.diag(jnp.linalg.inv(dense)),
    )
    diag = curvature.inverse_diagonal_estimate(op, overlap=0, block_size=1, correction_probes=0)
    selected = curvature.selected_inverse(op, overlap=0, block_size=1, correction_probes=0)

    expected = jnp.diag(jnp.linalg.inv(dense))
    assert jnp.allclose(diag, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(selected, expected, rtol=1e-6, atol=1e-6)


def test_selected_inverse_uses_solve_fallback_for_operator_only_curvature():
    dense = jnp.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    op = curvature.make_curvature_operator(
        shape=(2, 2),
        dtype=jnp.float64,
        matvec=lambda v: dense @ v,
        rmatvec=lambda v: dense.T @ v,
        solve_fn=lambda b, **kwargs: jnp.linalg.solve(dense, jnp.asarray(b, dtype=jnp.float64)),
        metadata={"symmetric": True, "psd": True},
    )

    idx = jnp.asarray([[0, 0], [0, 1], [1, 1]], dtype=jnp.int32)
    selected = curvature.selected_inverse(op, index_set=idx)
    inv = jnp.linalg.inv(dense)
    expected = inv[idx[:, 0], idx[:, 1]]
    assert jnp.allclose(selected, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(curvature.posterior_marginal_variances(op), jnp.diag(inv), rtol=1e-6, atol=1e-6)


def test_covariance_pushforward_supports_dense_and_operator_backed_curvature():
    dense = jnp.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    op_dense = curvature.make_dense_curvature_operator(dense, psd=True)
    op_shell = curvature.make_curvature_operator(
        shape=(2, 2),
        dtype=jnp.float64,
        matvec=lambda v: dense @ v,
        rmatvec=lambda v: dense.T @ v,
        solve_fn=lambda b, **kwargs: jnp.linalg.solve(dense, jnp.asarray(b, dtype=jnp.float64)),
        metadata={"symmetric": True, "psd": True},
    )
    linear = jnp.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=jnp.float64)
    cov = jnp.linalg.inv(dense)
    expected = linear @ cov @ linear.T

    assert jnp.allclose(curvature.covariance_pushforward(op_dense, linear), expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(curvature.covariance_pushforward(op_shell, linear), expected, rtol=1e-6, atol=1e-6)

    def map_fn(v):
        return linear @ v

    assert jnp.allclose(curvature.covariance_pushforward(op_shell, map_fn, output_dim=2), expected, rtol=1e-6, atol=1e-6)


def test_sparse_curvature_constructors_reuse_matrix_free_operator_surfaces():
    dense_r = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    sparse_r = sparse_common.SparseBCOO(
        data=jnp.asarray([2.0, 3.0], dtype=jnp.float64),
        indices=jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jrb",
    )
    probes_r = jnp.stack([_ivec2(1.0, 0.0), _ivec2(0.0, 1.0)], axis=0)
    op_r = curvature.make_jrb_sparse_curvature_operator(sparse_r, probes=probes_r, steps=2, symmetric=True)

    rhs_r = jnp.asarray([2.0, -3.0], dtype=jnp.float64)
    assert jnp.allclose(op_r.matvec(rhs_r), dense_r @ rhs_r, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(op_r.solve(rhs_r), jnp.asarray([1.0, -1.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(op_r.inverse_diagonal(), jnp.asarray([0.5, 1.0 / 3.0], dtype=jnp.float64), rtol=1e-6, atol=1e-6)

    sparse_c = sparse_common.SparseBCOO(
        data=jnp.asarray([2.0 + 0.0j, 5.0 + 0.0j], dtype=jnp.complex128),
        indices=jnp.asarray([[0, 0], [1, 1]], dtype=jnp.int32),
        rows=2,
        cols=2,
        algebra="jcb",
    )
    probes_c = jnp.stack([_cvec2(1.0 + 0.0j, 0.0 + 0.0j), _cvec2(0.0 + 0.0j, 1.0 + 0.0j)], axis=0)
    op_c = curvature.make_jcb_sparse_curvature_operator(sparse_c, probes=probes_c, steps=2, hermitian=True)

    rhs_c = jnp.asarray([2.0 + 0.0j, -5.0 + 0.0j], dtype=jnp.complex128)
    assert jnp.allclose(op_c.matvec(rhs_c), jnp.asarray([4.0 + 0.0j, -25.0 + 0.0j], dtype=jnp.complex128), rtol=1e-6, atol=1e-6)
    assert jnp.allclose(op_c.solve(rhs_c), jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j], dtype=jnp.complex128), rtol=1e-6, atol=1e-6)
