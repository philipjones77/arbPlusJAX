from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)

import argparse
import os
import time
from pathlib import Path

cache_dir = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        Path(__file__).resolve().parents[1] / "experiments" / "benchmarks" / "outputs" / "cache" / "jax_compilation_cache",
    )
).expanduser()
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import double_interval as di
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import precision
from arbplusjax import sparse_common

precision.enable_jax_x64()


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


def _time_compile_execute(fn, *args) -> tuple[float, float]:
    compile_s = _time_call(fn, *args)
    execute_s = _time_call(fn, *args)
    return compile_s, execute_s


def _time_warm_mean(fn, *args, warmup: int, runs: int) -> float:
    for _ in range(max(int(warmup), 0)):
        out = fn(*args)
        jax.block_until_ready(out)
    started = time.perf_counter()
    for _ in range(max(int(runs), 1)):
        out = fn(*args)
        jax.block_until_ready(out)
    elapsed = time.perf_counter() - started
    return elapsed / float(max(int(runs), 1))


def _precompile_many(entries: list[tuple[str, object, tuple]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, fn, args in entries:
        out[f"{name}_precompile_s"] = _time_call(fn, *args)
    return out


def run_real_case(n: int = 32, steps: int = 12, *, warmup: int = 2, runs: int = 5, precompile_hot: bool = True) -> dict[str, float]:
    a = _dense_real_diag(n)
    x = _real_vec(n)
    op = jrb_mat.jrb_mat_dense_operator(a)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(a)
    precond = jrb_mat.jrb_mat_jacobi_preconditioner_plan_prepare(plan)
    probes = jnp.stack([x, _real_vec(n)], axis=0)
    shifts = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)

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
    eigsh_fn = lambda: jrb_mat.jrb_mat_eigsh_point_jit(op, size=n, k=min(4, n), which="smallest", steps=min(n, max(steps, 2 * min(4, n))))
    eigsh_plan_fn = lambda: jrb_mat.jrb_mat_eigsh_point_jit(plan, size=n, k=min(4, n), which="smallest", steps=min(n, max(steps, 2 * min(4, n))))
    indef_diag = jax.vmap(_real_interval)(jnp.where((jnp.arange(n) % 2) == 0, jnp.linspace(2.0, 3.0, n), -jnp.linspace(2.0, 3.0, n)))
    indef = jnp.eye(n, dtype=jnp.float64)[..., None] * indef_diag[:, None, :]
    indef_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(indef)
    minres_rhs = jrb_mat.jrb_mat_operator_apply_point(indef_plan, x)
    minres_plan_fn = lambda vec: jrb_mat.jrb_mat_minres_solve_action_point_jit(indef_plan, vec, tol=1e-7, maxiter=max(steps * 4, n))
    multi_shift_plan_fn = lambda vec: jrb_mat.jrb_mat_multi_shift_solve_point_jit(
        plan,
        vec,
        shifts,
        symmetric=True,
        preconditioner=precond,
        tol=1e-7,
        maxiter=max(steps * 4, 8),
    )
    eigsh_restarted_plan_fn = lambda: jrb_mat.jrb_mat_eigsh_restarted_point_jit(
        plan,
        size=n,
        k=min(4, n),
        which="smallest",
        steps=min(max(2, steps // 2), n),
        restarts=2,
        block_size=min(max(4, min(4, n)), n),
    )
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jrb_mat.jrb_mat_logdet_slq_point(op, ps, steps))
    )
    solve_grad_fn = jax.jit(
        jax.grad(lambda vec: jnp.sum(di.midpoint(jrb_mat.jrb_mat_solve_action_point_jit(plan, vec, symmetric=True))))
    )
    inverse_grad_fn = jax.jit(
        jax.grad(lambda vec: jnp.sum(di.midpoint(jrb_mat.jrb_mat_inverse_action_point_jit(plan, vec, symmetric=True))))
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
    eigsh_time = _time_call(eigsh_fn)
    eigsh_plan_time = _time_call(eigsh_plan_fn)
    minres_plan_time = _time_call(minres_plan_fn, minres_rhs)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    solve_compile_time, solve_execute_time = _time_compile_execute(solve_plan_fn, x)
    inverse_compile_time, inverse_execute_time = _time_compile_execute(inverse_plan_fn, x)
    multi_shift_compile_time, multi_shift_execute_time = _time_compile_execute(multi_shift_plan_fn, x)
    eigsh_restart_compile_time, eigsh_restart_execute_time = _time_compile_execute(eigsh_restarted_plan_fn)
    minres_compile_time, minres_execute_time = _time_compile_execute(minres_plan_fn, minres_rhs)
    solve_grad_compile_time, solve_grad_execute_time = _time_compile_execute(solve_grad_fn, x)
    inverse_grad_compile_time, inverse_grad_execute_time = _time_compile_execute(inverse_grad_fn, x)
    logdet_grad_compile_time, logdet_grad_execute_time = _time_compile_execute(logdet_grad_fn, probes)
    precompile_stats = {}
    if precompile_hot:
        precompile_stats = _precompile_many([
            ("real_apply_plan", apply_plan_fn, (x,)),
            ("real_action_plan", plan_action_fn, (x,)),
            ("real_logdet_plan", logdet_plan_fn, (probes,)),
            ("real_det_plan", det_plan_fn, (probes,)),
        ])
    return {
        "real_apply_cold_s": apply_time,
        "real_apply_plan_cold_s": apply_plan_time,
        "real_action_cold_s": action_time,
        "real_action_plan_cold_s": action_plan_time,
        "real_restarted_action_cold_s": restarted_action_time,
        "real_restarted_action_plan_cold_s": restarted_action_plan_time,
        "real_grad_cold_s": grad_time,
        "real_logdet_cold_s": logdet_time,
        "real_logdet_plan_cold_s": logdet_plan_time,
        "real_det_cold_s": det_time,
        "real_det_plan_cold_s": det_plan_time,
        "real_solve_action_cold_s": solve_time,
        "real_solve_action_plan_cold_s": solve_plan_time,
        "real_inverse_action_cold_s": inverse_time,
        "real_inverse_action_plan_cold_s": inverse_plan_time,
        "real_eigsh_cold_s": eigsh_time,
        "real_eigsh_plan_cold_s": eigsh_plan_time,
        "real_minres_plan_cold_s": minres_plan_time,
        "real_logdet_grad_cold_s": logdet_grad_time,
        "real_solve_action_plan_compile_s": solve_compile_time,
        "real_solve_action_plan_execute_s": solve_execute_time,
        "real_inverse_action_plan_compile_s": inverse_compile_time,
        "real_inverse_action_plan_execute_s": inverse_execute_time,
        "real_multi_shift_plan_compile_s": multi_shift_compile_time,
        "real_multi_shift_plan_execute_s": multi_shift_execute_time,
        "real_eigsh_restarted_plan_compile_s": eigsh_restart_compile_time,
        "real_eigsh_restarted_plan_execute_s": eigsh_restart_execute_time,
        "real_minres_plan_compile_s": minres_compile_time,
        "real_minres_plan_execute_s": minres_execute_time,
        "real_solve_grad_plan_compile_s": solve_grad_compile_time,
        "real_solve_grad_plan_execute_s": solve_grad_execute_time,
        "real_inverse_grad_plan_compile_s": inverse_grad_compile_time,
        "real_inverse_grad_plan_execute_s": inverse_grad_execute_time,
        "real_logdet_grad_compile_s": logdet_grad_compile_time,
        "real_logdet_grad_execute_s": logdet_grad_execute_time,
        "real_apply_warm_s": _time_warm_mean(apply_fn, x, warmup=warmup, runs=runs),
        "real_apply_plan_warm_s": _time_warm_mean(apply_plan_fn, x, warmup=warmup, runs=runs),
        "real_action_warm_s": _time_warm_mean(action_fn, x, warmup=warmup, runs=runs),
        "real_action_plan_warm_s": _time_warm_mean(plan_action_fn, x, warmup=warmup, runs=runs),
        "real_logdet_warm_s": _time_warm_mean(logdet_fn, probes, warmup=warmup, runs=runs),
        "real_logdet_plan_warm_s": _time_warm_mean(logdet_plan_fn, probes, warmup=warmup, runs=runs),
        "real_solve_action_plan_warm_s": _time_warm_mean(solve_plan_fn, x, warmup=warmup, runs=runs),
        "real_inverse_action_plan_warm_s": _time_warm_mean(inverse_plan_fn, x, warmup=warmup, runs=runs),
        "real_multi_shift_plan_warm_s": _time_warm_mean(multi_shift_plan_fn, x, warmup=warmup, runs=runs),
        **precompile_stats,
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


def run_complex_case(n: int = 24, steps: int = 10, *, warmup: int = 2, runs: int = 5, precompile_hot: bool = True) -> dict[str, float]:
    a = _dense_complex_diag(n)
    x = _complex_vec(n)
    op = jcb_mat.jcb_mat_dense_operator(a)
    adj = jcb_mat.jcb_mat_dense_operator_adjoint(a)
    plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(a)
    aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(a)
    precond = jcb_mat.jcb_mat_jacobi_preconditioner_plan_prepare(plan)
    probes = jnp.stack([x, _complex_vec(n)], axis=0)
    shifts = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)

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
    solve_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(op, vec, hermitian=True)
    solve_plan_fn = lambda vec: jcb_mat.jcb_mat_solve_action_point_jit(plan, vec, hermitian=True)
    inverse_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(op, vec, hermitian=True)
    inverse_plan_fn = lambda vec: jcb_mat.jcb_mat_inverse_action_point_jit(plan, vec, hermitian=True)
    eigsh_fn = lambda: jcb_mat.jcb_mat_eigsh_point_jit(plan, size=n, k=min(4, n), which="largest", steps=min(n, max(steps, 2 * min(4, n))))
    eigsh_op_fn = lambda: jcb_mat.jcb_mat_eigsh_point_jit(op, size=n, k=min(4, n), which="largest", steps=min(n, max(steps, 2 * min(4, n))))
    indef_vals = jnp.where((jnp.arange(n) % 2) == 0, jnp.linspace(2.0, 3.0, n), -jnp.linspace(2.0, 3.0, n)).astype(jnp.complex128)
    indef = jnp.eye(n, dtype=jnp.complex128)
    indef = jax.vmap(_complex_box)(indef_vals)[:, None, :] * jnp.eye(n, dtype=jnp.float64)[..., None]
    indef_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(indef)
    minres_rhs = jcb_mat.jcb_mat_operator_apply_point(indef_plan, x)
    minres_plan_fn = lambda vec: jcb_mat.jcb_mat_minres_solve_action_point_jit(indef_plan, vec, tol=1e-7, maxiter=max(steps * 4, n))
    multi_shift_plan_fn = lambda vec: jcb_mat.jcb_mat_multi_shift_solve_point_jit(
        plan,
        vec,
        shifts,
        hermitian=True,
        preconditioner=precond,
        tol=1e-7,
        maxiter=max(steps * 4, 8),
    )
    eigsh_restarted_plan_fn = lambda: jcb_mat.jcb_mat_eigsh_restarted_point_jit(
        plan,
        size=n,
        k=min(4, n),
        which="largest",
        steps=min(max(2, steps // 2), n),
        restarts=2,
        block_size=min(max(4, min(4, n)), n),
    )
    logdet_grad_fn = jax.jit(
        jax.grad(lambda ps: jnp.real(jcb_mat.jcb_mat_logdet_slq_point(op, ps, steps, adj)))
    )
    solve_grad_fn = jax.jit(
        jax.grad(lambda vec: jnp.real(jnp.sum(acb_core.acb_midpoint(jcb_mat.jcb_mat_solve_action_point_jit(plan, vec, hermitian=True))))))
    inverse_grad_fn = jax.jit(
        jax.grad(lambda vec: jnp.real(jnp.sum(acb_core.acb_midpoint(jcb_mat.jcb_mat_inverse_action_point_jit(plan, vec, hermitian=True))))))

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
    eigsh_time = _time_call(eigsh_op_fn)
    eigsh_plan_time = _time_call(eigsh_fn)
    minres_plan_time = _time_call(minres_plan_fn, minres_rhs)
    logdet_grad_time = _time_call(logdet_grad_fn, probes)
    solve_compile_time, solve_execute_time = _time_compile_execute(solve_plan_fn, x)
    inverse_compile_time, inverse_execute_time = _time_compile_execute(inverse_plan_fn, x)
    multi_shift_compile_time, multi_shift_execute_time = _time_compile_execute(multi_shift_plan_fn, x)
    eigsh_restart_compile_time, eigsh_restart_execute_time = _time_compile_execute(eigsh_restarted_plan_fn)
    minres_compile_time, minres_execute_time = _time_compile_execute(minres_plan_fn, minres_rhs)
    solve_grad_compile_time, solve_grad_execute_time = _time_compile_execute(solve_grad_fn, x)
    inverse_grad_compile_time, inverse_grad_execute_time = _time_compile_execute(inverse_grad_fn, x)
    logdet_grad_compile_time, logdet_grad_execute_time = _time_compile_execute(logdet_grad_fn, probes)
    precompile_stats = {}
    if precompile_hot:
        precompile_stats = _precompile_many([
            ("complex_apply_plan", apply_plan_fn, (x,)),
            ("complex_action_plan", action_plan_fn, (x,)),
            ("complex_logdet_plan", logdet_plan_fn, (probes,)),
            ("complex_det_plan", det_plan_fn, (probes,)),
        ])
    return {
        "complex_apply_cold_s": apply_time,
        "complex_apply_plan_cold_s": apply_plan_time,
        "complex_action_cold_s": action_time,
        "complex_action_plan_cold_s": action_plan_time,
        "complex_restarted_action_cold_s": restarted_action_time,
        "complex_restarted_action_plan_cold_s": restarted_action_plan_time,
        "complex_grad_cold_s": grad_time,
        "complex_logdet_cold_s": logdet_time,
        "complex_logdet_plan_cold_s": logdet_plan_time,
        "complex_det_cold_s": det_time,
        "complex_det_plan_cold_s": det_plan_time,
        "complex_solve_action_cold_s": solve_time,
        "complex_solve_action_plan_cold_s": solve_plan_time,
        "complex_inverse_action_cold_s": inverse_time,
        "complex_inverse_action_plan_cold_s": inverse_plan_time,
        "complex_eigsh_cold_s": eigsh_time,
        "complex_eigsh_plan_cold_s": eigsh_plan_time,
        "complex_minres_plan_cold_s": minres_plan_time,
        "complex_logdet_grad_cold_s": logdet_grad_time,
        "complex_solve_action_plan_compile_s": solve_compile_time,
        "complex_solve_action_plan_execute_s": solve_execute_time,
        "complex_inverse_action_plan_compile_s": inverse_compile_time,
        "complex_inverse_action_plan_execute_s": inverse_execute_time,
        "complex_multi_shift_plan_compile_s": multi_shift_compile_time,
        "complex_multi_shift_plan_execute_s": multi_shift_execute_time,
        "complex_eigsh_restarted_plan_compile_s": eigsh_restart_compile_time,
        "complex_eigsh_restarted_plan_execute_s": eigsh_restart_execute_time,
        "complex_minres_plan_compile_s": minres_compile_time,
        "complex_minres_plan_execute_s": minres_execute_time,
        "complex_solve_grad_plan_compile_s": solve_grad_compile_time,
        "complex_solve_grad_plan_execute_s": solve_grad_execute_time,
        "complex_inverse_grad_plan_compile_s": inverse_grad_compile_time,
        "complex_inverse_grad_plan_execute_s": inverse_grad_execute_time,
        "complex_logdet_grad_compile_s": logdet_grad_compile_time,
        "complex_logdet_grad_execute_s": logdet_grad_execute_time,
        "complex_apply_plan_warm_s": _time_warm_mean(apply_plan_fn, x, warmup=warmup, runs=runs),
        "complex_action_plan_warm_s": _time_warm_mean(action_plan_fn, x, warmup=warmup, runs=runs),
        "complex_logdet_plan_warm_s": _time_warm_mean(logdet_plan_fn, probes, warmup=warmup, runs=runs),
        "complex_solve_action_plan_warm_s": _time_warm_mean(solve_plan_fn, x, warmup=warmup, runs=runs),
        "complex_inverse_action_plan_warm_s": _time_warm_mean(inverse_plan_fn, x, warmup=warmup, runs=runs),
        "complex_multi_shift_plan_warm_s": _time_warm_mean(multi_shift_plan_fn, x, warmup=warmup, runs=runs),
        **precompile_stats,
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


def _parse_sections(raw: str | None) -> set[str]:
    if raw is None:
        return {"real", "sparse_real", "complex", "sparse_complex"}
    out = {part.strip() for part in raw.split(",") if part.strip()}
    return out or {"real", "sparse_real", "complex", "sparse_complex"}


def main():
    parser = argparse.ArgumentParser(description="Matrix-free Krylov benchmark with cold/warm timing separation.")
    parser.add_argument("--n-real", type=int, default=32)
    parser.add_argument("--n-complex", type=int, default=24)
    parser.add_argument("--steps-real", type=int, default=12)
    parser.add_argument("--steps-complex", type=int, default=10)
    parser.add_argument("--steps-sparse", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--no-plan-precompile", action="store_true")
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Comma-separated subset of sections to run: real,sparse_real,complex,sparse_complex",
    )
    args = parser.parse_args()
    sections = _parse_sections(args.sections)

    results: dict[str, float] = {}
    if "real" in sections:
        print("[matrix_free_krylov] running real", flush=True)
        results.update(
            run_real_case(
                args.n_real,
                args.steps_real,
                warmup=args.warmup,
                runs=args.runs,
                precompile_hot=not args.no_plan_precompile,
            )
        )
    if "sparse_real" in sections:
        print("[matrix_free_krylov] running sparse_real", flush=True)
        results.update(run_sparse_real_parametric_case(args.steps_sparse))
    if "complex" in sections:
        print("[matrix_free_krylov] running complex", flush=True)
        results.update(
            run_complex_case(
                args.n_complex,
                args.steps_complex,
                warmup=args.warmup,
                runs=args.runs,
                precompile_hot=not args.no_plan_precompile,
            )
        )
    if "sparse_complex" in sections:
        print("[matrix_free_krylov] running sparse_complex", flush=True)
        results.update(run_sparse_complex_case())
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"plan_precompile: {not args.no_plan_precompile}")
    print(f"sections: {','.join(sorted(sections))}")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
