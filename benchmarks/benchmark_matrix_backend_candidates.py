from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time
from typing import Any

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np

di = None
jrb_mat = None
mat_wrappers = None
srb_mat = None


def _load_matrix_backend_modules() -> None:
    global di, jrb_mat, mat_wrappers, srb_mat
    if di is None:
        from arbplusjax import double_interval as _di
        from arbplusjax import jrb_mat as _jrb_mat
        from arbplusjax import mat_wrappers as _mat_wrappers
        from arbplusjax import srb_mat as _srb_mat

        di = _di
        jrb_mat = _jrb_mat
        mat_wrappers = _mat_wrappers
        srb_mat = _srb_mat

try:
    import scipy.linalg as scipy_linalg
    import scipy.sparse as scipy_sparse
    import scipy.sparse.linalg as scipy_sparse_linalg

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from petsc4py import PETSc

    HAS_PETSC = True
except Exception:
    HAS_PETSC = False

try:
    from slepc4py import SLEPc

    HAS_SLEPC = True
except Exception:
    HAS_SLEPC = False


def _block(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            _block(item)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time_call(fn, *args, warmup: int, runs: int) -> float:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        _block(out)
    started = time.perf_counter()
    for _ in range(runs):
        out = fn(*args)
        _block(out)
    ended = time.perf_counter()
    return (ended - started) / float(runs)


def _spd_dense_case(n: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    _load_matrix_backend_modules()
    base = jnp.reshape(jnp.linspace(0.2, 1.2, n * n, dtype=jnp.float64), (n, n))
    dense = base.T @ base + jnp.eye(n, dtype=jnp.float64) * (n + 1.0)
    rhs = jnp.linspace(-0.5, 0.75, n, dtype=jnp.float64)
    probes = jnp.stack([rhs, rhs + 1.0], axis=0)
    return dense, rhs, probes


def _sparse_case(n: int, density: float) -> tuple[jax.Array, Any, Any, Any, jax.Array]:
    _load_matrix_backend_modules()
    dense, rhs, probes = _spd_dense_case(n)
    dense_np = np.asarray(dense)
    mask = np.zeros((n, n), dtype=bool)
    bandwidth = max(1, int(round(max(1.0, density * n))))
    for offset in range(-bandwidth, bandwidth + 1):
        mask |= np.eye(n, k=offset, dtype=bool)
    mask |= np.eye(n, dtype=bool)
    sparse_np = np.where(mask, dense_np, 0.0)
    sparse_np = 0.5 * (sparse_np + sparse_np.T)
    sparse_np += np.eye(n) * (n + 1.0)
    dense_jax = jnp.asarray(sparse_np, dtype=jnp.float64)
    srb = srb_mat.srb_mat_from_dense_bcoo(dense_jax)
    bcoo = jsparse.BCOO.fromdense(dense_jax)
    scipy_csr = scipy_sparse.csr_matrix(sparse_np) if HAS_SCIPY else None
    return dense_jax, srb, bcoo, scipy_csr, rhs


def _petsc_aij_from_scipy(csr) -> Any:
    mat = PETSc.Mat().createAIJ(size=csr.shape, csr=(csr.indptr, csr.indices, csr.data))
    mat.assemble()
    return mat


def _petsc_vec_from_numpy(x: np.ndarray) -> Any:
    vec = PETSc.Vec().createSeq(len(x))
    vec.setValues(range(len(x)), x)
    vec.assemble()
    return vec


def run_candidate_suite(n: int, warmup: int, runs: int, density: float) -> dict[str, float]:
    dense, rhs, probes = _spd_dense_case(n)
    dense_np = np.asarray(dense)
    rhs_np = np.asarray(rhs)
    _, srb_bcoo, js_bcoo, scipy_csr, sparse_rhs = _sparse_case(n, density)

    results: dict[str, float] = {}

    jax_dense_matvec = jax.jit(lambda a, x: a @ x)
    jax_dense_solve = jax.jit(lambda a, b: jnp.linalg.solve(a, b))
    jax_dense_eigh = jax.jit(jnp.linalg.eigh)
    jax_scipy_dense_solve = jax.jit(lambda a, b: jsp_linalg.solve(a, b, assume_a="pos"))

    results["candidate_jax_dense_matvec_s"] = _time_call(jax_dense_matvec, dense, rhs, warmup=warmup, runs=runs)
    results["candidate_jax_dense_solve_s"] = _time_call(jax_dense_solve, dense, rhs, warmup=warmup, runs=runs)
    results["candidate_jax_dense_eigh_s"] = _time_call(jax_dense_eigh, dense, warmup=warmup, runs=runs)
    results["candidate_jax_scipy_dense_solve_s"] = _time_call(
        jax_scipy_dense_solve, dense, rhs, warmup=warmup, runs=runs
    )

    arb_dense = lambda a, b: mat_wrappers.srb_mat_spd_solve_mode(srb_mat.srb_mat_from_dense_bcoo(a), b, impl="point")
    results["candidate_arbplusjax_sparse_fromdense_solve_s"] = _time_call(
        arb_dense, dense, rhs, warmup=warmup, runs=runs
    )

    srb_cached = mat_wrappers.srb_mat_matvec_cached_prepare_mode(srb_bcoo, impl="point")
    srb_spd_plan = mat_wrappers.srb_mat_spd_solve_plan_prepare_mode(srb_bcoo, impl="point")
    results["candidate_arbplusjax_sparse_matvec_s"] = _time_call(
        lambda a, x: mat_wrappers.srb_mat_matvec_mode(a, x, impl="point"),
        srb_bcoo,
        sparse_rhs,
        warmup=warmup,
        runs=runs,
    )
    results["candidate_arbplusjax_sparse_cached_matvec_s"] = _time_call(
        lambda plan, x: mat_wrappers.srb_mat_matvec_cached_apply_mode(plan, x, impl="point"),
        srb_cached,
        sparse_rhs,
        warmup=warmup,
        runs=runs,
    )
    results["candidate_arbplusjax_sparse_spd_solve_s"] = _time_call(
        lambda plan, b: mat_wrappers.srb_mat_spd_solve_plan_apply_mode(plan, b, impl="point"),
        srb_spd_plan,
        sparse_rhs,
        warmup=warmup,
        runs=runs,
    )

    js_matvec = jax.jit(lambda a, x: a @ x)
    js_cg = jax.jit(lambda a, b: jsp_sparse_linalg.cg(lambda v: a @ v, b, tol=1e-8, maxiter=max(16, n))[0])
    results["candidate_jax_experimental_sparse_matvec_s"] = _time_call(
        js_matvec, js_bcoo, sparse_rhs, warmup=warmup, runs=runs
    )
    results["candidate_jax_experimental_sparse_cg_s"] = _time_call(js_cg, js_bcoo, sparse_rhs, warmup=warmup, runs=runs)

    interval_dense = di.interval(dense, dense)
    interval_rhs = di.interval(rhs, rhs)
    op = jrb_mat.jrb_mat_dense_operator(interval_dense)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(interval_dense)
    results["candidate_matfree_apply_s"] = _time_call(
        lambda p, x: jrb_mat.jrb_mat_operator_plan_apply(p, x), plan, interval_rhs, warmup=warmup, runs=runs
    )
    results["candidate_matfree_solve_action_s"] = _time_call(
        lambda p, x: jrb_mat.jrb_mat_solve_action_point_jit(p, x, symmetric=True),
        plan,
        interval_rhs,
        warmup=warmup,
        runs=runs,
    )
    results["candidate_matfree_logdet_slq_s"] = _time_call(
        lambda operator, ps: jrb_mat.jrb_mat_logdet_slq_point_jit(operator, ps, min(n, 8)),
        op,
        jnp.stack([interval_rhs, interval_rhs], axis=0),
        warmup=warmup,
        runs=runs,
    )

    if HAS_SCIPY and scipy_csr is not None:
        results["candidate_scipy_dense_matvec_s"] = _time_call(
            lambda a, x: a @ x, dense_np, rhs_np, warmup=warmup, runs=runs
        )
        results["candidate_scipy_dense_solve_s"] = _time_call(
            scipy_linalg.solve, dense_np, rhs_np, warmup=warmup, runs=runs
        )
        results["candidate_scipy_dense_eigh_s"] = _time_call(
            scipy_linalg.eigh, dense_np, warmup=warmup, runs=runs
        )
        results["candidate_scipy_sparse_matvec_s"] = _time_call(
            lambda a, x: a @ x, scipy_csr, rhs_np, warmup=warmup, runs=runs
        )
        results["candidate_scipy_sparse_cg_s"] = _time_call(
            lambda a, b: scipy_sparse_linalg.cg(a, b, rtol=1e-8, maxiter=max(16, n))[0],
            scipy_csr,
            rhs_np,
            warmup=warmup,
            runs=runs,
        )
        results["candidate_scipy_sparse_eigsh_s"] = _time_call(
            lambda a: scipy_sparse_linalg.eigsh(a, k=min(2, n - 1), which="SM", return_eigenvectors=False),
            scipy_csr,
            warmup=warmup,
            runs=runs,
        )
        linop = scipy_sparse_linalg.LinearOperator(
            shape=scipy_csr.shape,
            matvec=lambda x: scipy_csr @ x,
            dtype=scipy_csr.dtype,
        )
        results["candidate_scipy_linear_operator_cg_s"] = _time_call(
            lambda a, b: scipy_sparse_linalg.cg(a, b, rtol=1e-8, maxiter=max(16, n))[0],
            linop,
            rhs_np,
            warmup=warmup,
            runs=runs,
        )

    if HAS_PETSC:
        results["candidate_petsc_available"] = 1.0
        if HAS_SCIPY and scipy_csr is not None:
            try:
                petsc_mat = _petsc_aij_from_scipy(scipy_csr)
                petsc_x = _petsc_vec_from_numpy(rhs_np)
                petsc_y = PETSc.Vec().createSeq(len(rhs_np))
                results["candidate_petsc_matvec_s"] = _time_call(
                    lambda a, x, y: a.mult(x, y), petsc_mat, petsc_x, petsc_y, warmup=warmup, runs=runs
                )
                ksp = PETSc.KSP().create()
                ksp.setOperators(petsc_mat)
                ksp.setType("cg")
                ksp.getPC().setType("jacobi")
                ksp.setTolerances(rtol=1e-8, max_it=max(16, n))
                results["candidate_petsc_ksp_cg_s"] = _time_call(
                    lambda solver, b, x: solver.solve(b, x),
                    ksp,
                    petsc_x,
                    petsc_y,
                    warmup=warmup,
                    runs=runs,
                )
            except Exception:
                results["candidate_petsc_runtime_ok"] = 0.0
    else:
        results["candidate_petsc_available"] = 0.0

    if HAS_SLEPC:
        results["candidate_slepc_available"] = 1.0
        if HAS_PETSC and HAS_SCIPY and scipy_csr is not None:
            try:
                slepc_mat = _petsc_aij_from_scipy(scipy_csr)
                eps = SLEPc.EPS().create()
                eps.setOperators(slepc_mat)
                eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
                eps.setDimensions(nev=min(2, n - 1))
                eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
                results["candidate_slepc_eps_s"] = _time_call(eps.solve, warmup=warmup, runs=runs)
            except Exception:
                results["candidate_slepc_runtime_ok"] = 0.0
    else:
        results["candidate_slepc_available"] = 0.0

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark candidate matrix backends across dense, sparse, and matrix-free surfaces."
    )
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n: {args.n}")
    print(f"density: {args.density}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"has_scipy: {HAS_SCIPY}")
    print(f"has_petsc: {HAS_PETSC}")
    print(f"has_slepc: {HAS_SLEPC}")

    stats = run_candidate_suite(args.n, args.warmup, args.runs, args.density)
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
