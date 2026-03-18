import time

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import sparse_common


def _real_interval(x: jax.Array) -> jax.Array:
    return di.interval(x, x)


def _complex_box(z: jax.Array) -> jax.Array:
    return acb_core.acb_box(_real_interval(jnp.real(z)), _real_interval(jnp.imag(z)))


def _dense_real_diag(n: int) -> jax.Array:
    vals = jnp.linspace(1.0, 2.0, n, dtype=jnp.float64)
    diag = jax.vmap(_real_interval)(vals)
    return jnp.eye(n, dtype=jnp.float64)[..., None] * diag[:, None, :]


def _dense_complex_diag(n: int) -> jax.Array:
    vals = jnp.linspace(1.0, 2.0, n, dtype=jnp.float64) + 0.25j * jnp.linspace(0.0, 1.0, n, dtype=jnp.float64)
    diag = jax.vmap(_complex_box)(vals)
    return jnp.eye(n, dtype=jnp.float64)[..., None] * diag[:, None, :]


def _real_vec(n: int) -> jax.Array:
    vals = jnp.linspace(0.5, 1.5, n, dtype=jnp.float64)
    return jax.vmap(_real_interval)(vals)


def _complex_vec(n: int) -> jax.Array:
    vals = jnp.linspace(0.5, 1.5, n, dtype=jnp.float64) + 0.1j * jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)
    return jax.vmap(_complex_box)(vals)


def _dense_exp_sym(matrix: jax.Array) -> jax.Array:
    vals, vecs = jnp.linalg.eigh(matrix)
    return vecs @ jnp.diag(jnp.exp(vals)) @ vecs.T


def _dense_exp_general(matrix: jax.Array) -> jax.Array:
    vals, vecs = jnp.linalg.eig(matrix)
    return vecs @ jnp.diag(jnp.exp(vals)) @ jnp.linalg.inv(vecs)


def _time_call(fn, *args):
    started = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return time.perf_counter() - started


def run_real_case(n: int = 32, steps: int = 12) -> dict[str, float]:
    a = _dense_real_diag(n)
    x = _real_vec(n)
    op = jrb_mat.jrb_mat_dense_operator(a)
    probes = jnp.stack([x, _real_vec(n)], axis=0)

    def loss(vec):
        y = jrb_mat.jrb_mat_funm_action_lanczos_point(op, vec, _dense_exp_sym, steps)
        return jnp.sum(di.midpoint(y))

    grad_fn = jax.jit(jax.grad(loss))
    action_fn = jax.jit(lambda vec: jrb_mat.jrb_mat_funm_action_lanczos_point(op, vec, _dense_exp_sym, steps))
    restarted_action_fn = jax.jit(
        lambda vec: jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(op, vec, steps=steps, restarts=2)
    )
    logdet_fn = jax.jit(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(op, ps, steps))
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(op, ps, steps))
    )

    action_time = _time_call(action_fn, x)
    restarted_action_time = _time_call(restarted_action_fn, x)
    grad_time = _time_call(grad_fn, x)
    logdet_time = _time_call(logdet_fn, probes)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    return {
        "real_action_s": action_time,
        "real_restarted_action_s": restarted_action_time,
        "real_grad_s": grad_time,
        "real_logdet_s": logdet_time,
        "real_logdet_grad_s": logdet_grad_time,
    }


def run_sparse_real_parametric_case(steps: int = 12) -> dict[str, float]:
    steps = min(int(steps), 4)
    indices = jnp.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ],
        dtype=jnp.int32,
    )
    base_data = jnp.asarray([2.0, -0.3, -0.3, 2.5, -0.25, -0.25, 3.0, -0.2, -0.2, 3.5], dtype=jnp.float64)
    probes = jnp.stack([
        jax.vmap(_real_interval)(2.0 * jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)),
        jax.vmap(_real_interval)(2.0 * jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)),
        jax.vmap(_real_interval)(2.0 * jnp.asarray([0.0, 0.0, 1.0, 0.0], dtype=jnp.float64)),
        jax.vmap(_real_interval)(2.0 * jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)),
    ], axis=0)
    bcoo = sparse_common.SparseBCOO(data=base_data, indices=indices, rows=4, cols=4, algebra="jrb")
    fixed_op = jrb_mat.jrb_mat_bcoo_operator(bcoo)
    bounds = jrb_mat.jrb_mat_bcoo_gershgorin_bounds(bcoo)
    sketch = probes
    residual = jnp.zeros((0, 4, 2), dtype=jnp.float64)

    logdet_sparse_fn = jax.jit(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(fixed_op, ps, steps))
    logdet_leja_fn = jax.jit(
        lambda ss, rs: jrb_mat.jrb_mat_logdet_leja_hutchpp_point(
            fixed_op,
            ss,
            rs,
            degree=max(steps, 6),
            spectral_bounds=bounds,
        )
    )
    logdet_leja_auto_fn = jax.jit(
        lambda ss, rs: jrb_mat.jrb_mat_bcoo_logdet_leja_hutchpp_point(
            bcoo,
            ss,
            rs,
            degree=max(steps, 6),
            max_degree=max(steps + 8, 12),
            min_degree=4,
        )
    )
    logdet_sparse_grad_fn = jax.jit(jax.grad(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(fixed_op, ps, steps)))
    apply_fn = jax.jit(lambda x: jrb_mat.jrb_mat_operator_apply_point(fixed_op, x))
    inverse_diag_local_fn = lambda: jrb_mat.jrb_mat_bcoo_inverse_diagonal_point(
        bcoo,
        overlap=0,
        block_size=2,
        correction_probes=0,
    )
    inverse_diag_corrected_fn = lambda: jrb_mat.jrb_mat_bcoo_inverse_diagonal_point(
        bcoo,
        overlap=0,
        block_size=2,
        correction_probes=32,
        key=jax.random.PRNGKey(0),
        tol=1e-7,
        maxiter=16,
    )
    vec = jax.vmap(_real_interval)(jnp.asarray([1.0, -1.0, 0.5, 2.0], dtype=jnp.float64))

    apply_time = _time_call(apply_fn, vec)
    logdet_time = _time_call(logdet_sparse_fn, probes)
    logdet_leja_time = _time_call(logdet_leja_fn, sketch, residual)
    logdet_leja_auto_time = _time_call(logdet_leja_auto_fn, sketch, residual)
    logdet_grad_time = _time_call(logdet_sparse_grad_fn, probes)
    inverse_diag_local_time = _time_call(inverse_diag_local_fn)
    inverse_diag_corrected_time = _time_call(inverse_diag_corrected_fn)
    return {
        "sparse_real_apply_s": apply_time,
        "sparse_real_logdet_s": logdet_time,
        "sparse_real_logdet_leja_hutchpp_s": logdet_leja_time,
        "sparse_real_logdet_leja_hutchpp_auto_s": logdet_leja_auto_time,
        "sparse_real_logdet_grad_s": logdet_grad_time,
        "sparse_real_inverse_diag_local_s": inverse_diag_local_time,
        "sparse_real_inverse_diag_corrected_s": inverse_diag_corrected_time,
    }


def run_complex_case(n: int = 24, steps: int = 10) -> dict[str, float]:
    a = _dense_complex_diag(n)
    x = _complex_vec(n)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)
    probes = jnp.stack([x, _complex_vec(n)], axis=0)

    def loss(vec):
        y = jcb_mat.jcb_mat_funm_action_arnoldi_point(op, vec, _dense_exp_general, steps, adj)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    grad_fn = jax.jit(jax.grad(loss))
    action_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_funm_action_arnoldi_point(op, vec, _dense_exp_general, steps, adj))
    restarted_action_fn = jax.jit(
        lambda vec: jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(
            op,
            vec,
            steps=steps,
            restarts=2,
            adjoint_matvec=adj,
        )
    )
    logdet_fn = jax.jit(lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point(op, ps, steps, adj)))
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point(op, ps, steps, adj)))
    )

    action_time = _time_call(action_fn, x)
    restarted_action_time = _time_call(restarted_action_fn, x)
    grad_time = _time_call(grad_fn, x)
    logdet_time = _time_call(logdet_fn, probes)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    return {
        "complex_action_s": action_time,
        "complex_restarted_action_s": restarted_action_time,
        "complex_grad_s": grad_time,
        "complex_logdet_s": logdet_time,
        "complex_logdet_grad_s": logdet_grad_time,
    }


def main():
    real = run_real_case()
    sparse_real = run_sparse_real_parametric_case()
    complex_case = run_complex_case()
    for key, value in {**real, **sparse_real, **complex_case}.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
