import jax
import jax.numpy as jnp

from arbplusjax import curvature
from arbplusjax import double_interval as di
from arbplusjax import jrb_mat


def _imat2(a00, a01, a10, a11):
    row0 = jnp.stack([di.interval(a00, a00), di.interval(a01, a01)], axis=0)
    row1 = jnp.stack([di.interval(a10, a10), di.interval(a11, a11)], axis=0)
    return jnp.stack([row0, row1], axis=0)


def _ivec2(x0, x1):
    xs = jnp.asarray([x0, x1], dtype=jnp.float64)
    return di.interval(xs, xs)


def test_dense_curvature_fast_jax_surface_supports_jit_and_vmap():
    dense = jnp.asarray([[3.0, 0.5], [0.5, 2.0]], dtype=jnp.float64)
    op = curvature.make_dense_curvature_operator(dense, psd=True)

    xs = jnp.asarray([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]], dtype=jnp.float64)
    matvec_jit = jax.jit(jax.vmap(op.matvec))
    solve_jit = jax.jit(jax.vmap(op.solve))
    logdet_jit = jax.jit(lambda: op.logdet())
    invdiag_jit = jax.jit(lambda: op.inverse_diagonal())

    assert jnp.allclose(matvec_jit(xs), xs @ dense.T)
    assert jnp.allclose(solve_jit(xs), jnp.linalg.solve(dense, xs.T).T)
    assert jnp.allclose(logdet_jit(), jnp.linalg.slogdet(dense)[1])
    assert jnp.allclose(invdiag_jit(), jnp.diag(jnp.linalg.inv(dense)))


def test_curvature_parameter_ad_for_posterior_damping_and_jitter_is_jittable():
    prior_dense = jnp.diag(jnp.asarray([2.0, 5.0], dtype=jnp.float64))
    like_dense = jnp.diag(jnp.asarray([1.5, 0.5], dtype=jnp.float64))
    prior = curvature.make_dense_curvature_operator(prior_dense, psd=True)
    like = curvature.make_dense_curvature_operator(like_dense, psd=True)

    def objective(damping, jitter):
        post = curvature.make_posterior_precision_operator(
            prior,
            like,
            damping=damping,
            jitter=jitter,
        )
        return post.logdet()

    grad_fn = jax.jit(jax.grad(lambda d, j: objective(d, j), argnums=(0, 1)))
    damping = jnp.asarray(0.25, dtype=jnp.float64)
    jitter = jnp.asarray(0.125, dtype=jnp.float64)
    g_damping, g_jitter = grad_fn(damping, jitter)

    eigs = jnp.asarray([3.875, 5.875], dtype=jnp.float64)
    expected = jnp.sum(1.0 / eigs)
    assert jnp.allclose(g_damping, expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(g_jitter, expected, rtol=1e-6, atol=1e-6)


def test_curvature_variable_and_parameter_ad_for_hvp_surface():
    def loss(theta, x):
        scale = jnp.asarray([theta + 1.0, 2.0 * theta + 0.5], dtype=jnp.float64)
        return 0.5 * jnp.dot(x, scale * x)

    x = jnp.asarray([1.25, -0.5], dtype=jnp.float64)
    v = jnp.asarray([0.75, 2.0], dtype=jnp.float64)
    theta = jnp.asarray(1.5, dtype=jnp.float64)

    hvp_theta = jax.jit(lambda t: curvature.hvp(lambda z: loss(t, z), x, v))
    assert jnp.allclose(hvp_theta(theta), jnp.asarray([1.875, 7.0], dtype=jnp.float64))

    scalarized = lambda t, z: jnp.sum(curvature.hvp(lambda w: loss(t, w), z, v))
    grad_x = jax.jit(jax.grad(lambda z: scalarized(theta, z)))(x)
    grad_theta = jax.jit(jax.grad(lambda t: scalarized(t, x)))(theta)

    assert jnp.allclose(grad_x, jnp.zeros_like(x))
    assert jnp.allclose(grad_theta, jnp.sum(v * jnp.asarray([1.0, 2.0], dtype=jnp.float64)))


def test_matrix_free_curvature_operational_surface_supports_jitted_reuse():
    dense = _imat2(2.0, 0.0, 0.0, 3.0)
    plan = jrb_mat.jrb_mat_dense_operator(dense)
    probes = jnp.stack([_ivec2(1.0, 1.0), _ivec2(1.0, -1.0)], axis=0)
    op = curvature.make_jrb_curvature_operator(plan, shape=(2, 2), probes=probes, steps=2, symmetric=True)

    xs = jnp.asarray([[2.0, -3.0], [1.0, 6.0]], dtype=jnp.float64)
    matvec_jit = jax.jit(jax.vmap(op.matvec))
    solve_jit = jax.jit(jax.vmap(op.solve))
    logdet_jit = jax.jit(lambda: op.logdet())

    expected_dense = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    assert jnp.allclose(matvec_jit(xs), xs @ expected_dense.T, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(solve_jit(xs), jnp.linalg.solve(expected_dense, xs.T).T, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(logdet_jit(), jnp.log(2.0) + jnp.log(3.0), rtol=1e-6, atol=1e-6)
