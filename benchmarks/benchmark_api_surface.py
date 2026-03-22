from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time

import jax
import jax.numpy as jnp

from arbplusjax import acb_core
from arbplusjax import acb_mat
from arbplusjax import api
from arbplusjax import arb_mat
from arbplusjax import double_interval as di


def _interval_matrix_from_point(a: jax.Array) -> jax.Array:
    return di.interval(a, a)


def _box_matrix_from_point(a: jax.Array) -> jax.Array:
    return acb_core.acb_box(
        di.interval(jnp.real(a), jnp.real(a)),
        di.interval(jnp.imag(a), jnp.imag(a)),
    )


def _time_call(fn, *args, warmup: int, runs: int) -> float:
    out = None
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    started = time.perf_counter()
    for _ in range(runs):
        out = fn(*args)
        jax.block_until_ready(out)
    ended = time.perf_counter()
    return (ended - started) / float(runs)


def run_scalar_case(warmup: int, runs: int) -> dict[str, float]:
    x = jnp.asarray(0.5, dtype=jnp.float64)
    y = jnp.asarray(2.0, dtype=jnp.float64)
    direct = jax.jit(lambda a, b: api.eval_point("cuda_besselk", a, b))
    routed = jax.jit(lambda a, b: api.evaluate("besselk", a, b, implementation="cuda_besselk", value_kind="real"))
    return {
        "api_scalar_direct_cuda_besselk_s": _time_call(direct, x, y, warmup=warmup, runs=runs),
        "api_scalar_routed_cuda_besselk_s": _time_call(routed, x, y, warmup=warmup, runs=runs),
    }


def run_incomplete_gamma_case(warmup: int, runs: int) -> dict[str, float]:
    s = jnp.asarray(2.5, dtype=jnp.float64)
    z = jnp.asarray(1.0, dtype=jnp.float64)
    direct = jax.jit(
        lambda a, b: api.incomplete_gamma_upper(a, b, method="quadrature", samples_per_panel=8, max_panels=16)
    )
    routed = jax.jit(
        lambda a, b: api.evaluate(
            "incomplete_gamma_upper",
            a,
            b,
            method="quadrature",
            method_params={"samples_per_panel": 8, "max_panels": 16},
        )
    )
    return {
        "api_incgamma_direct_s": _time_call(direct, s, z, warmup=warmup, runs=runs),
        "api_incgamma_routed_s": _time_call(routed, s, z, warmup=warmup, runs=runs),
    }


def run_matrix_case(warmup: int, runs: int) -> dict[str, float]:
    a_mid = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    rhs_mid = jnp.array([[1.0], [2.0]], dtype=jnp.float64)
    a = _interval_matrix_from_point(a_mid)
    rhs = _interval_matrix_from_point(rhs_mid)

    c_mid = jnp.array([[4.0 + 0.0j, 1.0 + 1.0j], [1.0 - 1.0j, 5.0 + 0.0j]], dtype=jnp.complex128)
    c_rhs_mid = jnp.array([[1.0 + 0.5j], [2.0 - 0.25j]], dtype=jnp.complex128)
    c_a = _box_matrix_from_point(c_mid)
    c_rhs = _box_matrix_from_point(c_rhs_mid)

    direct_real = jax.jit(lambda aa, bb: api.eval_interval("arb_mat_solve", aa, bb, mode="basic"))
    routed_real = jax.jit(
        lambda aa, bb: api.evaluate("arb_mat_solve", aa, bb, mode="basic", value_kind="real_interval_matrix")
    )
    direct_complex = jax.jit(lambda aa, bb: api.eval_interval("acb_mat_solve", aa, bb, mode="basic"))
    routed_complex = jax.jit(
        lambda aa, bb: api.evaluate("acb_mat_solve", aa, bb, mode="basic", value_kind="complex_interval_matrix")
    )
    return {
        "api_matrix_real_direct_s": _time_call(direct_real, a, rhs, warmup=warmup, runs=runs),
        "api_matrix_real_routed_s": _time_call(routed_real, a, rhs, warmup=warmup, runs=runs),
        "api_matrix_complex_direct_s": _time_call(direct_complex, c_a, c_rhs, warmup=warmup, runs=runs),
        "api_matrix_complex_routed_s": _time_call(routed_complex, c_a, c_rhs, warmup=warmup, runs=runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark routed public API overhead against direct API calls.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")

    stats = {}
    stats.update(run_scalar_case(args.warmup, args.runs))
    stats.update(run_incomplete_gamma_case(args.warmup, args.runs))
    stats.update(run_matrix_case(args.warmup, args.runs))
    for key in sorted(stats):
        print(f"{key}: {stats[key]:.6e}")


if __name__ == "__main__":
    main()
