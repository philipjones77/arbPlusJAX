from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import acb_mat
from arbplusjax import acb_core
from arbplusjax import arb_mat
from arbplusjax import double_interval as di
from arbplusjax import jax_diagnostics
from arbplusjax import jcb_mat
from arbplusjax import jrb_mat
from arbplusjax import scb_mat
from arbplusjax import sparse_common
from arbplusjax import srb_mat


def _interval(x):
    return di.interval(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))


def _box(z):
    return acb_core.acb_box(_interval(jnp.real(z)), _interval(jnp.imag(z)))


def _dense_real_matrix(n: int):
    mid = jnp.arange(1, n * n + 1, dtype=jnp.float64).reshape(n, n) / float(n)
    return di.interval(mid, mid)


def _dense_real_vector(n: int):
    mid = jnp.linspace(1.0, float(n), n, dtype=jnp.float64)
    return di.interval(mid, mid)


def _dense_complex_matrix(n: int):
    re = jnp.arange(1, n * n + 1, dtype=jnp.float64).reshape(n, n) / float(n)
    im = jnp.flip(re, axis=-1) / float(n + 1)
    return acb_core.acb_box(di.interval(re, re), di.interval(im, im))


def _dense_complex_vector(n: int):
    re = jnp.linspace(1.0, float(n), n, dtype=jnp.float64)
    im = jnp.linspace(0.5, float(n) - 0.5, n, dtype=jnp.float64)
    return acb_core.acb_box(di.interval(re, re), di.interval(im, im))


def _diag_sparse_real(n: int):
    idx = jnp.stack([jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32)], axis=1)
    data = jnp.linspace(2.0, float(n + 1), n, dtype=jnp.float64)
    return sparse_common.SparseBCOO(data=data, indices=idx, rows=n, cols=n, algebra="jrb")


def _diag_sparse_complex(n: int):
    idx = jnp.stack([jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32)], axis=1)
    data = jnp.linspace(2.0, float(n + 1), n, dtype=jnp.float64) + 0.25j
    return sparse_common.SparseBCOO(data=data.astype(jnp.complex128), indices=idx, rows=n, cols=n, algebra="jcb")


def build_cases(n: int):
    dense_r = _dense_real_matrix(n)
    dense_c = _dense_complex_matrix(n)
    vec_r = _dense_real_vector(n)
    vec_c = _dense_complex_vector(n)
    sparse_r = _diag_sparse_real(n)
    sparse_c = _diag_sparse_complex(n)
    probes_r = jnp.stack([vec_r, vec_r], axis=0)
    probes_c = jnp.stack([vec_c, vec_c], axis=0)

    dense_r_cache = arb_mat.arb_mat_matvec_cached_prepare(dense_r)
    dense_c_cache = acb_mat.acb_mat_matvec_cached_prepare(dense_c)
    dense_r_plan = jrb_mat.jrb_mat_dense_operator_plan_prepare(dense_r)
    dense_c_plan = jcb_mat.jcb_mat_dense_operator_plan_prepare(dense_c)
    dense_c_aplan = jcb_mat.jcb_mat_dense_operator_adjoint_plan_prepare(dense_c)
    sparse_r_plan = jrb_mat.jrb_mat_bcoo_operator_plan_prepare(sparse_r)
    sparse_c_plan = jcb_mat.jcb_mat_bcoo_operator_plan_prepare(sparse_c)
    sparse_c_aplan = jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(sparse_c)

    return [
        {
            "name": "arb_dense_matvec_cached_apply",
            "fn": jax.jit(lambda payload: arb_mat.arb_mat_matvec_cached_apply(payload[0], payload[1])),
            "arg": (dense_r_cache, vec_r),
            "alt_arg": (arb_mat.arb_mat_matvec_cached_prepare(_dense_real_matrix(n + 1)), _dense_real_vector(n + 1)),
        },
        {
            "name": "acb_dense_matvec_cached_apply",
            "fn": jax.jit(lambda payload: acb_mat.acb_mat_matvec_cached_apply(payload[0], payload[1])),
            "arg": (dense_c_cache, vec_c),
            "alt_arg": (acb_mat.acb_mat_matvec_cached_prepare(_dense_complex_matrix(n + 1)), _dense_complex_vector(n + 1)),
        },
        {
            "name": "srb_sparse_matvec_point",
            "fn": jax.jit(lambda payload: srb_mat.srb_mat_matvec(payload[0], payload[1])),
            "arg": (srb_mat.srb_mat_from_dense_bcoo(di.midpoint(dense_r)), di.midpoint(vec_r)),
            "alt_arg": (srb_mat.srb_mat_from_dense_bcoo(di.midpoint(_dense_real_matrix(n + 1))), di.midpoint(_dense_real_vector(n + 1))),
        },
        {
            "name": "jrb_operator_apply_point",
            "fn": jax.jit(lambda payload: jrb_mat.jrb_mat_operator_apply_point(payload[0], payload[1])),
            "arg": (dense_r_plan, vec_r),
            "alt_arg": (jrb_mat.jrb_mat_dense_operator_plan_prepare(_dense_real_matrix(n + 1)), _dense_real_vector(n + 1)),
        },
        {
            "name": "jrb_logdet_slq_point",
            "fn": lambda payload: jrb_mat.jrb_mat_logdet_slq_point_jit(payload[0], payload[1], steps=min(8, n)),
            "arg": (sparse_r_plan, probes_r),
            "alt_arg": (jrb_mat.jrb_mat_bcoo_operator_plan_prepare(_diag_sparse_real(n + 1)), jnp.stack([_dense_real_vector(n + 1), _dense_real_vector(n + 1)], axis=0)),
        },
        {
            "name": "jcb_operator_apply_point",
            "fn": jax.jit(lambda payload: jcb_mat.jcb_mat_operator_apply_point(payload[0], payload[1])),
            "arg": (dense_c_plan, vec_c),
            "alt_arg": (jcb_mat.jcb_mat_dense_operator_plan_prepare(_dense_complex_matrix(n + 1)), _dense_complex_vector(n + 1)),
        },
        {
            "name": "jcb_logdet_slq_point",
            "fn": lambda payload: jcb_mat.jcb_mat_logdet_slq_point_jit(payload[0], payload[1], steps=min(8, n), adjoint_matvec=payload[2]),
            "arg": (dense_c_plan, probes_c, dense_c_aplan),
            "alt_arg": (
                jcb_mat.jcb_mat_bcoo_operator_plan_prepare(_diag_sparse_complex(n + 1)),
                jnp.stack([_dense_complex_vector(n + 1), _dense_complex_vector(n + 1)], axis=0),
                jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(_diag_sparse_complex(n + 1)),
            ),
        },
        {
            "name": "jcb_sparse_logdet_slq_point",
            "fn": lambda payload: jcb_mat.jcb_mat_logdet_slq_point_jit(payload[0], payload[1], steps=min(8, n), adjoint_matvec=payload[2]),
            "arg": (sparse_c_plan, probes_c, sparse_c_aplan),
            "alt_arg": (
                jcb_mat.jcb_mat_bcoo_operator_plan_prepare(_diag_sparse_complex(n + 1)),
                jnp.stack([_dense_complex_vector(n + 1), _dense_complex_vector(n + 1)], axis=0),
                jcb_mat.jcb_mat_bcoo_operator_adjoint_plan_prepare(_diag_sparse_complex(n + 1)),
            ),
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Profile representative dense/sparse/matrix-free JAX matrix kernels.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/diagnostics/matrix_stack_profile.json"),
    )
    args = parser.parse_args()

    cfg = jax_diagnostics.config_from_env()
    profiles = jax_diagnostics.profile_function_suite(build_cases(args.n), repeats=args.repeats, config=cfg)
    path = jax_diagnostics.write_profile_report(args.output, profiles)
    print(path)
    for profile in profiles:
        print(
            f"{profile.name}: compile={profile.compile_ms:.2f}ms "
            f"steady_med={profile.steady_ms_median:.2f}ms "
            f"recompile={profile.recompile_new_shape_ms:.2f}ms "
            f"rss_delta={profile.peak_rss_delta_mb:.2f}MB "
            f"device_delta={profile.device_memory_delta_mb if profile.device_memory_delta_mb is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()
