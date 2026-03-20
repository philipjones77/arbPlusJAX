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
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    probes = jnp.stack([x, _real_vec(n)], axis=0)

    def loss(vec):
        y = jrb_mat.jrb_mat_funm_action_lanczos_point(op, vec, _dense_exp_sym, steps)
        return jnp.sum(di.midpoint(y))

    grad_fn = jax.jit(jax.grad(loss))
    action_fn = jax.jit(lambda vec: jrb_mat.jrb_mat_funm_action_lanczos_point(op, vec, _dense_exp_sym, steps))
    plan_action_fn = jax.jit(lambda vec: jrb_mat.jrb_mat_funm_action_lanczos_point(plan, vec, _dense_exp_sym, steps))
    restarted_action_fn = jax.jit(
        lambda vec: jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(op, vec, steps=steps, restarts=2)
    )
    restarted_plan_fn = jax.jit(
        lambda vec: jrb_mat.jrb_mat_expm_action_lanczos_restarted_point(plan, vec, steps=steps, restarts=2)
    )
    apply_fn = jax.jit(lambda vec: jrb_mat.jrb_mat_operator_apply_point(op, vec))
    apply_plan_fn = jax.jit(lambda vec: jrb_mat.jrb_mat_operator_plan_apply(plan, vec))
    logdet_fn = lambda ps: jrb_mat.jrb_mat_logdet_slq_point_jit(op, ps, steps)
    logdet_plan_fn = lambda ps: jrb_mat.jrb_mat_logdet_slq_point_jit(plan, ps, steps)
    det_fn = lambda ps: jrb_mat.jrb_mat_det_slq_point_jit(op, ps, steps)
    det_plan_fn = lambda ps: jrb_mat.jrb_mat_det_slq_point_jit(plan, ps, steps)
    solve_fn = lambda vec: jrb_mat.jrb_mat_solve_action_point_jit(op, vec, symmetric=True)
    solve_plan_fn = lambda vec: jrb_mat.jrb_mat_solve_action_point_jit(plan, vec, symmetric=True)
    inverse_fn = lambda vec: jrb_mat.jrb_mat_inverse_action_point_jit(op, vec, symmetric=True)
    inverse_plan_fn = lambda vec: jrb_mat.jrb_mat_inverse_action_point_jit(plan, vec, symmetric=True)
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(op, ps, steps))
    )

    apply_time = _time_call(apply_fn, x)
    apply_plan_time = _time_call(apply_plan_fn, x)
    action_time = _time_call(action_fn, x)
    action_plan_time = _time_call(plan_action_fn, x)
    restarted_action_time = _time_call(restarted_action_fn, x)
    restarted_action_plan_time = _time_call(restarted_plan_fn, x)
    grad_time = _time_call(grad_fn, x)
    logdet_time = _time_call(logdet_fn, probes)
    logdet_plan_time = _time_call(logdet_plan_fn, probes)
    det_time = _time_call(det_fn, probes)
    det_plan_time = _time_call(det_plan_fn, probes)
    solve_time = _time_call(solve_fn, x)
    solve_plan_time = _time_call(solve_plan_fn, x)
    inverse_time = _time_call(inverse_fn, x)
    inverse_plan_time = _time_call(inverse_plan_fn, x)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    return {
        "real_apply_s": apply_time,
        "real_apply_plan_s": apply_plan_time,
        "real_action_s": action_time,
        "real_action_plan_s": action_plan_time,
        "real_restarted_action_s": restarted_action_time,
        "real_restarted_action_plan_s": restarted_action_plan_time,
        "real_grad_s": grad_time,
        "real_logdet_s": logdet_time,
        "real_logdet_plan_s": logdet_plan_time,
        "real_det_s": det_time,
        "real_det_plan_s": det_plan_time,
        "real_solve_action_s": solve_time,
        "real_solve_action_plan_s": solve_plan_time,
        "real_inverse_action_s": inverse_time,
        "real_inverse_action_plan_s": inverse_plan_time,
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
    fixed_plan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(bcoo)
    bounds = jrb_mat.jrb_mat_bcoo_gershgorin_bounds(bcoo)
    sketch = probes
    residual = jnp.zeros((0, 4, 2), dtype=jnp.float64)

    logdet_sparse_fn = lambda ps: jrb_mat.jrb_mat_logdet_slq_point_jit(fixed_op, ps, steps)
    logdet_sparse_plan_fn = lambda ps: jrb_mat.jrb_mat_logdet_slq_point_jit(fixed_plan, ps, steps)
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
    det_sparse_fn = lambda ps: jrb_mat.jrb_mat_det_slq_point_jit(fixed_op, ps, steps)
    det_sparse_plan_fn = lambda ps: jrb_mat.jrb_mat_det_slq_point_jit(fixed_plan, ps, steps)
    solve_sparse_fn = lambda x: jrb_mat.jrb_mat_solve_action_point_jit(fixed_op, x, symmetric=True, tol=1e-7, maxiter=16)
    solve_sparse_plan_fn = lambda x: jrb_mat.jrb_mat_solve_action_point_jit(fixed_plan, x, symmetric=True, tol=1e-7, maxiter=16)
    inverse_sparse_fn = lambda x: jrb_mat.jrb_mat_inverse_action_point_jit(fixed_op, x, symmetric=True, tol=1e-7, maxiter=16)
    inverse_sparse_plan_fn = lambda x: jrb_mat.jrb_mat_inverse_action_point_jit(fixed_plan, x, symmetric=True, tol=1e-7, maxiter=16)
    apply_fn = jax.jit(lambda x: jrb_mat.jrb_mat_operator_apply_point(fixed_op, x))
    apply_plan_fn = jax.jit(lambda x: jrb_mat.jrb_mat_operator_plan_apply(fixed_plan, x))
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
    apply_plan_time = _time_call(apply_plan_fn, vec)
    logdet_time = _time_call(logdet_sparse_fn, probes)
    logdet_plan_time = _time_call(logdet_sparse_plan_fn, probes)
    logdet_leja_time = _time_call(logdet_leja_fn, sketch, residual)
    logdet_leja_auto_time = _time_call(logdet_leja_auto_fn, sketch, residual)
    logdet_grad_time = _time_call(logdet_sparse_grad_fn, probes)
    det_time = _time_call(det_sparse_fn, probes)
    det_plan_time = _time_call(det_sparse_plan_fn, probes)
    solve_time = _time_call(solve_sparse_fn, vec)
    solve_plan_time = _time_call(solve_sparse_plan_fn, vec)
    inverse_time = _time_call(inverse_sparse_fn, vec)
    inverse_plan_time = _time_call(inverse_sparse_plan_fn, vec)
    inverse_diag_local_time = _time_call(inverse_diag_local_fn)
    inverse_diag_corrected_time = _time_call(inverse_diag_corrected_fn)
    return {
        "sparse_real_apply_s": apply_time,
        "sparse_real_apply_plan_s": apply_plan_time,
        "sparse_real_logdet_s": logdet_time,
        "sparse_real_logdet_plan_s": logdet_plan_time,
        "sparse_real_det_s": det_time,
        "sparse_real_det_plan_s": det_plan_time,
        "sparse_real_solve_action_s": solve_time,
        "sparse_real_solve_action_plan_s": solve_plan_time,
        "sparse_real_inverse_action_s": inverse_time,
        "sparse_real_inverse_action_plan_s": inverse_plan_time,
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
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    probes = jnp.stack([x, _complex_vec(n)], axis=0)

    def loss(vec):
        y = jcb_mat.jcb_mat_funm_action_arnoldi_point(op, vec, _dense_exp_general, steps, adj)
        return jnp.real(jnp.sum(acb_core.acb_midpoint(y)))

    grad_fn = jax.jit(jax.grad(loss))
    action_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_funm_action_arnoldi_point(op, vec, _dense_exp_general, steps, adj))
    action_plan_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_funm_action_arnoldi_point(plan, vec, _dense_exp_general, steps, aplan))
    restarted_action_fn = jax.jit(
        lambda vec: jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(
            op,
            vec,
            steps=steps,
            restarts=2,
            adjoint_matvec=adj,
        )
    )
    restarted_action_plan_fn = jax.jit(
        lambda vec: jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(
            plan,
            vec,
            steps=steps,
            restarts=2,
            adjoint_matvec=aplan,
        )
    )
    apply_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_operator_apply_point(op, vec))
    apply_plan_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_operator_plan_apply(plan, vec))
    logdet_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point_jit(op, ps, steps, adj))
    logdet_plan_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point_jit(plan, ps, steps, aplan))
    det_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_det_slq_point_jit(op, ps, steps, adj))
    det_plan_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_det_slq_point_jit(plan, ps, steps, aplan))
    solve_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(op, vec)
    solve_plan_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(plan, vec)
    inverse_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(op, vec)
    inverse_plan_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(plan, vec)
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point(op, ps, steps, adj)))
    )

    apply_time = _time_call(apply_fn, x)
    apply_plan_time = _time_call(apply_plan_fn, x)
    action_time = _time_call(action_fn, x)
    action_plan_time = _time_call(action_plan_fn, x)
    restarted_action_time = _time_call(restarted_action_fn, x)
    restarted_action_plan_time = _time_call(restarted_action_plan_fn, x)
    grad_time = _time_call(grad_fn, x)
    logdet_time = _time_call(logdet_fn, probes)
    logdet_plan_time = _time_call(logdet_plan_fn, probes)
    det_time = _time_call(det_fn, probes)
    det_plan_time = _time_call(det_plan_fn, probes)
    solve_time = _time_call(solve_fn, x)
    solve_plan_time = _time_call(solve_plan_fn, x)
    inverse_time = _time_call(inverse_fn, x)
    inverse_plan_time = _time_call(inverse_plan_fn, x)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    return {
        "complex_apply_s": apply_time,
        "complex_apply_plan_s": apply_plan_time,
        "complex_action_s": action_time,
        "complex_action_plan_s": action_plan_time,
        "complex_restarted_action_s": restarted_action_time,
        "complex_restarted_action_plan_s": restarted_action_plan_time,
        "complex_grad_s": grad_time,
        "complex_logdet_s": logdet_time,
        "complex_logdet_plan_s": logdet_plan_time,
        "complex_det_s": det_time,
        "complex_det_plan_s": det_plan_time,
        "complex_solve_action_s": solve_time,
        "complex_solve_action_plan_s": solve_plan_time,
        "complex_inverse_action_s": inverse_time,
        "complex_inverse_action_plan_s": inverse_plan_time,
        "complex_logdet_grad_s": logdet_grad_time,
    }


def run_sparse_complex_case(steps: int = 8) -> dict[str, float]:
    steps = min(int(steps), 2)
    indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    data = jnp.asarray([2.0 + 0.0j, 1.0 + 1.0j, 1.0 - 1.0j, 3.0 + 0.0j], dtype=jnp.complex128)
    bcoo = sparse_common.SparseBCOO(data=data, indices=indices, rows=2, cols=2, algebra="jcb")
    op = jcb_mat.jcb_mat_bcoo_operator(bcoo)
    adj = jcb_mat.jcb_mat_bcoo_operator_adjoint(bcoo)
    plan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(bcoo)
    aplan = jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(bcoo)
    x = _complex_vec(2)
    probes = jnp.stack([x, _complex_vec(2)], axis=0)

    apply_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_operator_apply_point(op, vec))
    apply_plan_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_operator_plan_apply(plan, vec))
    action_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_funm_action_arnoldi_point(op, vec, _dense_exp_general, steps, adj))
    action_plan_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_funm_action_arnoldi_point(plan, vec, _dense_exp_general, steps, aplan))
    restarted_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(op, vec, steps=steps, restarts=2, adjoint_matvec=adj))
    restarted_plan_fn = jax.jit(lambda vec: jcb_mat.jcb_mat_expm_action_arnoldi_restarted_point(plan, vec, steps=steps, restarts=2, adjoint_matvec=aplan))
    logdet_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point_jit(op, ps, steps, adj))
    logdet_plan_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point_jit(plan, ps, steps, aplan))
    det_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_det_slq_point_jit(op, ps, steps, adj))
    det_plan_fn = lambda ps: jnp.real(jcb_mat.jcb_mat_det_slq_point_jit(plan, ps, steps, aplan))
    solve_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(op, vec, hermitian=True, tol=1e-7, maxiter=16)
    solve_plan_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(plan, vec, hermitian=True, tol=1e-7, maxiter=16)
    inverse_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(op, vec, hermitian=True, tol=1e-7, maxiter=16)
    inverse_plan_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(plan, vec, hermitian=True, tol=1e-7, maxiter=16)

    return {
        "sparse_complex_apply_s": _time_call(apply_fn, x),
        "sparse_complex_apply_plan_s": _time_call(apply_plan_fn, x),
        "sparse_complex_action_s": _time_call(action_fn, x),
        "sparse_complex_action_plan_s": _time_call(action_plan_fn, x),
        "sparse_complex_restarted_s": _time_call(restarted_fn, x),
        "sparse_complex_restarted_plan_s": _time_call(restarted_plan_fn, x),
        "sparse_complex_logdet_s": _time_call(logdet_fn, probes),
        "sparse_complex_logdet_plan_s": _time_call(logdet_plan_fn, probes),
        "sparse_complex_det_s": _time_call(det_fn, probes),
        "sparse_complex_det_plan_s": _time_call(det_plan_fn, probes),
        "sparse_complex_solve_action_s": _time_call(solve_fn, x),
        "sparse_complex_solve_action_plan_s": _time_call(solve_plan_fn, x),
        "sparse_complex_inverse_action_s": _time_call(inverse_fn, x),
        "sparse_complex_inverse_action_plan_s": _time_call(inverse_plan_fn, x),
    }


def main():
    real = run_real_case()
    sparse_real = run_sparse_real_parametric_case()
    complex_case = run_complex_case()
    sparse_complex = run_sparse_complex_case()
    for key, value in {**real, **sparse_real, **complex_case, **sparse_complex}.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
