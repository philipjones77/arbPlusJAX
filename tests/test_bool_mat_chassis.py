import jax
import jax.numpy as jnp

from arbplusjax import bool_mat


from tests._test_checks import _check
def test_jit_compiles():
    a = jnp.array([[1, 0, 0, 1], [1, 1, 0, 1]], dtype=jnp.uint8)
    det = bool_mat.bool_mat_2x2_det_batch_jit(a)
    trace = bool_mat.bool_mat_2x2_trace_batch_jit(a)
    _check(det.shape == (2,))
    _check(trace.shape == (2,))


def test_grad_path():
    def loss(t):
        mat = jnp.array([1, 0, 0, 1], dtype=jnp.uint8)
        return bool_mat.bool_mat_2x2_det(mat).astype(jnp.float64) + 0.0 * t

    g = jax.grad(loss)(jnp.float64(0.25))
    _check(bool(jnp.isfinite(g)))
