from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _time_call(fn, *args, iters: int = 1):
    import jax

    started = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return out, (time.perf_counter() - started) / float(iters)


def _bench_core_scalar_pow(iters: int) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from arbplusjax import api

    x = jnp.float32(1.3)
    y = jnp.float32(0.7)
    argument_grad = jax.grad(lambda xv: api.eval_point("arb_pow", xv, y, dtype="float32"))
    parameter_grad = jax.grad(lambda yv: api.eval_point("arb_pow", x, yv, dtype="float32"))
    _, argument_s = _time_call(argument_grad, x, iters=iters)
    _, parameter_s = _time_call(parameter_grad, y, iters=iters)
    return {"argument_grad_s": round(argument_s, 6), "parameter_grad_s": round(parameter_s, 6)}


def _bench_dense_matrix(iters: int) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from arbplusjax import double_interval as di
    from arbplusjax import jrb_mat

    base = jnp.array([[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64)
    vec_mid = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float64)
    vec = di.interval(vec_mid, vec_mid)

    def loss_vec(v):
        plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(base, base))
        out = jrb_mat.jrb_mat_operator_plan_apply(plan, di.interval(v, v))
        return jnp.sum(di.midpoint(out))

    def loss_scale(s):
        plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(s * base, s * base))
        out = jrb_mat.jrb_mat_operator_plan_apply(plan, vec)
        return jnp.sum(di.midpoint(out))

    argument_grad = jax.grad(loss_vec)
    parameter_grad = jax.grad(loss_scale)
    _, argument_s = _time_call(argument_grad, vec_mid, iters=iters)
    _, parameter_s = _time_call(parameter_grad, jnp.float64(1.0), iters=iters)
    return {"argument_grad_s": round(argument_s, 6), "parameter_grad_s": round(parameter_s, 6)}


def _bench_sparse_matrix(iters: int) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from arbplusjax import api
    from arbplusjax import srb_mat

    base = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]], dtype=jnp.float64)
    vec = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)

    def loss_vec(v):
        sparse = srb_mat.srb_mat_from_dense_bcoo(base)
        out = api.eval_point("srb_mat_matvec", sparse, v)
        return jnp.sum(out)

    def loss_scale(s):
        sparse = srb_mat.srb_mat_from_dense_bcoo(s * base)
        out = api.eval_point("srb_mat_matvec", sparse, vec)
        return jnp.sum(out)

    argument_grad = jax.grad(loss_vec)
    parameter_grad = jax.grad(loss_scale)
    _, argument_s = _time_call(argument_grad, vec, iters=iters)
    _, parameter_s = _time_call(parameter_grad, jnp.float64(1.0), iters=iters)
    return {"argument_grad_s": round(argument_s, 6), "parameter_grad_s": round(parameter_s, 6)}


def _bench_matrix_free_operator(iters: int) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from arbplusjax import double_interval as di
    from arbplusjax import jrb_mat

    base_diag = jnp.array([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)
    a_mid = jnp.diag(base_diag)
    plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(di.interval(a_mid, a_mid))
    rhs_mid = jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float64)
    rhs = di.interval(rhs_mid, rhs_mid)

    def loss_rhs(v):
        solved = jrb_mat.jrb_mat_solve_action_point_jit(plan, di.interval(v, v), symmetric=True)
        return jnp.sum(di.midpoint(solved))

    def loss_shift(s):
        solved = jrb_mat.jrb_mat_multi_shift_solve_point(plan, rhs, jnp.asarray([s], dtype=jnp.float64), symmetric=True)
        return jnp.sum(di.midpoint(solved))

    argument_grad = jax.grad(loss_rhs)
    parameter_grad = jax.grad(loss_shift)
    _, argument_s = _time_call(argument_grad, rhs_mid, iters=iters)
    _, parameter_s = _time_call(parameter_grad, jnp.float64(0.2), iters=iters)
    return {"argument_grad_s": round(argument_s, 6), "parameter_grad_s": round(parameter_s, 6)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark argument-direction and parameter-direction AD for representative public scalar, matrix, sparse, and operator surfaces."
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--out-json",
        default="benchmarks/results/public_surface_ad_benchmark/public_surface_ad_benchmark.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmarks/results/public_surface_ad_benchmark/public_surface_ad_benchmark.md",
    )
    args = parser.parse_args()

    payload = {
        "core_scalar_arb_pow": _bench_core_scalar_pow(args.iters),
        "dense_matrix_operator_apply": _bench_dense_matrix(args.iters),
        "sparse_matrix_matvec": _bench_sparse_matrix(args.iters),
        "matrix_free_multi_shift_solve": _bench_matrix_free_operator(args.iters),
    }

    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Public Surface AD Benchmark",
        "",
        "Generated by `benchmarks/public_surface_ad_benchmark.py`.",
        "",
    ]
    for name, values in payload.items():
        lines.extend(
            [
                f"## {name}",
                f"- argument grad s: `{values['argument_grad_s']}`",
                f"- parameter grad s: `{values['parameter_grad_s']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
