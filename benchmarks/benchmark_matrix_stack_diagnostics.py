from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

acb_mat = None
acb_core = None
arb_mat = None
di = None
jax_diagnostics = None
jcb_mat = None
jrb_mat = None
scb_mat = None
sparse_common = None
srb_mat = None


def _load_matrix_stack_modules() -> None:
    global acb_mat, acb_core, arb_mat, di, jax_diagnostics, jcb_mat, jrb_mat, scb_mat, sparse_common, srb_mat
    if acb_mat is None:
        from arbplusjax import acb_core as _acb_core
        from arbplusjax import acb_mat as _acb_mat
        from arbplusjax import arb_mat as _arb_mat
        from arbplusjax import double_interval as _di
        from arbplusjax import jax_diagnostics as _jax_diagnostics
        from arbplusjax import jcb_mat as _jcb_mat
        from arbplusjax import jrb_mat as _jrb_mat
        from arbplusjax import scb_mat as _scb_mat
        from arbplusjax import sparse_common as _sparse_common
        from arbplusjax import srb_mat as _srb_mat

        acb_core = _acb_core
        acb_mat = _acb_mat
        arb_mat = _arb_mat
        di = _di
        jax_diagnostics = _jax_diagnostics
        jcb_mat = _jcb_mat
        jrb_mat = _jrb_mat
        scb_mat = _scb_mat
        sparse_common = _sparse_common
        srb_mat = _srb_mat


def _interval(x):
    _load_matrix_stack_modules()
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
    _load_matrix_stack_modules()
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


def _parse_case_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    out = {part.strip() for part in raw.split(",") if part.strip()}
    return out or None


def main():
    parser = argparse.ArgumentParser(description="Profile representative dense/sparse/matrix-free JAX matrix kernels.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated subset of case names to run.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/diagnostics/matrix_stack_profile.json"),
    )
    args = parser.parse_args()
    _load_matrix_stack_modules()

    cfg = jax_diagnostics.config_from_env()
    cases = build_cases(args.n)
    requested = _parse_case_filter(args.cases)
    if requested is not None:
        cases = [case for case in cases if case["name"] in requested]
    print(
        "[matrix_stack_diagnostics] cases:",
        ",".join(case["name"] for case in cases),
        flush=True,
    )
    profiles = jax_diagnostics.profile_function_suite(cases, repeats=args.repeats, config=cfg)
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
