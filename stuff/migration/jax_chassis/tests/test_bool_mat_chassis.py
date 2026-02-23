import jax
import jax.numpy as jnp

from arbjax import bool_mat


def test_jit_compiles():
    a = jnp.array([[1, 0, 0, 1], [1, 1, 0, 1]], dtype=jnp.uint8)
    det = bool_mat.bool_mat_2x2_det_batch_jit(a)
    trace = bool_mat.bool_mat_2x2_trace_batch_jit(a)
    assert det.shape == (2,)
    assert trace.shape == (2,)


def test_grad_path():
    def loss(t):
        mat = jnp.array([1, 0, 0, 1], dtype=jnp.uint8)
        return bool_mat.bool_mat_2x2_det(mat).astype(jnp.float64) + 0.0 * t

    g = jax.grad(loss)(jnp.float64(0.25))
    assert bool(jnp.isfinite(g))
