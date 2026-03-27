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


def _bench_incomplete_bessel_i(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import api

    nu = jnp.asarray([12.0, 13.0, 14.0, 15.0], dtype=jnp.float64)
    z = jnp.asarray([9.0, 9.5, 10.0, 10.5], dtype=jnp.float64)
    upper = jnp.asarray([jnp.pi - 0.05, jnp.pi - 0.04, jnp.pi - 0.03, jnp.pi - 0.02], dtype=jnp.float64)

    quadrature, quadrature_s = _time_call(
        lambda a, b, c: api.incomplete_bessel_i_batch(a, b, c, mode="point", method="quadrature"),
        nu,
        z,
        upper,
        iters=iters,
    )
    refine, refine_s = _time_call(
        lambda a, b, c: api.incomplete_bessel_i_batch(a, b, c, mode="point", method="high_precision_refine"),
        nu,
        z,
        upper,
        iters=iters,
    )
    auto, auto_s = _time_call(
        lambda a, b, c: api.incomplete_bessel_i_batch(a, b, c, mode="point", method="auto"),
        nu,
        z,
        upper,
        iters=iters,
    )
    return {
        "quadrature_s": round(quadrature_s, 6),
        "high_precision_refine_s": round(refine_s, 6),
        "auto_s": round(auto_s, 6),
        "refine_vs_quadrature_max_abs": round(float(jnp.max(jnp.abs(refine - quadrature))), 6),
        "auto_vs_refine_max_abs": round(float(jnp.max(jnp.abs(auto - refine))), 6),
    }


def _bench_barnes_double_gamma(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import barnesg
    from arbplusjax import double_gamma
    from arbplusjax import stable_kernels

    z = jnp.asarray(0.7 + 0.1j, dtype=jnp.complex128)
    tau = jnp.asarray(1.0, dtype=jnp.float64)
    zs = jnp.asarray([0.7 + 0.1j, 1.2 + 0.15j, 1.7 + 0.2j, 2.2 + 0.25j], dtype=jnp.complex128)

    ifj_vals, ifj_s = _time_call(lambda arr: double_gamma.ifj_barnesdoublegamma(arr, tau, dps=60), zs, iters=iters)
    provider_vals, provider_s = _time_call(lambda arr: stable_kernels.barnesdoublegamma(arr, tau, dps=60), zs, iters=iters)
    bdg_vals, bdg_s = _time_call(lambda arr: double_gamma.bdg_barnesdoublegamma_batch_fixed_point(arr, jnp.full(arr.shape, tau), prec_bits=80), zs, iters=iters)

    recurrence_lhs = double_gamma.ifj_log_barnesdoublegamma(z + tau, tau, dps=60) - double_gamma.ifj_log_barnesdoublegamma(z, tau, dps=60)
    recurrence_rhs = barnesg._complex_loggamma(z)
    diagnostics = double_gamma.ifj_barnesdoublegamma_diagnostics(z, tau, dps=60, max_m_cap=96)

    return {
        "ifj_vector_s": round(ifj_s, 6),
        "provider_vector_s": round(provider_s, 6),
        "bdg_vector_s": round(bdg_s, 6),
        "ifj_shift_recurrence_abs": round(float(jnp.abs(recurrence_lhs - recurrence_rhs)), 12),
        "provider_vs_ifj_vector_max_abs": round(float(jnp.max(jnp.abs(provider_vals - ifj_vals))), 6),
        "ifj_vs_bdg_vector_max_abs": round(float(jnp.max(jnp.abs(ifj_vals - bdg_vals))), 6),
        "ifj_diag_m_used": int(diagnostics.m_used),
        "ifj_diag_m_capped": bool(diagnostics.m_capped),
        "ifj_diag_n_shift": int(diagnostics.n_shift),
    }


def _bench_hypgeom(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import acb_core
    from arbplusjax import api
    from arbplusjax import double_interval as di
    from arbplusjax import hypgeom
    from arbplusjax import hypgeom_wrappers

    def exact(x):
        xx = jnp.asarray(x, dtype=jnp.result_type(x, jnp.float64))
        return di.interval(xx, xx)

    a = jnp.asarray([1.0, 1.25, 1.5, 1.75], dtype=jnp.float64)
    b = jnp.asarray([2.0, 2.25, 2.5, 2.75], dtype=jnp.float64)
    z = jnp.asarray([0.2, 0.3, 0.4, 0.5], dtype=jnp.float64)
    bound = api.bind_point_batch_jit("hypgeom.arb_hypgeom_1f1", dtype="float64", pad_to=8)
    _, onef1_s = _time_call(lambda aa, bb, zz: bound(aa, bb, zz), a, b, z, iters=iters)
    u_bound = api.bind_point_batch_jit("hypgeom.arb_hypgeom_u", dtype="float64", pad_to=8)
    u_a = jnp.asarray([1.1, 1.2, 1.3, 1.4], dtype=jnp.float64)
    u_b = jnp.asarray([2.1, 2.2, 2.3, 2.4], dtype=jnp.float64)
    u_z = jnp.asarray([0.6, 0.8, 1.0, 1.2], dtype=jnp.float64)
    u_point, u_point_s = _time_call(lambda aa, bb, zz: u_bound(aa, bb, zz), u_a, u_b, u_z, iters=iters)
    u_mode, u_mode_s = _time_call(
        lambda aa, bb, zz: hypgeom_wrappers.arb_hypgeom_u_batch_mode_padded(
            di.interval(aa, aa),
            di.interval(bb, bb),
            di.interval(zz, zz),
            pad_to=8,
            impl="adaptive",
            prec_bits=53,
        ),
        u_a,
        u_b,
        u_z,
        iters=iters,
    )
    u_family = jnp.asarray(
        [
            di.midpoint(
                hypgeom.arb_hypgeom_u(
                    exact(aa),
                    exact(bb),
                    exact(zz),
                )
            )
            for aa, bb, zz in zip(u_a, u_b, u_z, strict=True)
        ],
        dtype=jnp.float64,
    )

    pfq_a = jnp.asarray(
        [[0.6, 0.9], [0.7, 1.0], [0.8, 1.1], [0.9, 1.2]],
        dtype=jnp.float64,
    )
    pfq_b = jnp.asarray([[1.4], [1.5], [1.6], [1.7]], dtype=jnp.float64)
    pfq_z = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64)
    pfq_bound = api.bind_point_batch_jit("hypgeom.arb_hypgeom_pfq", dtype="float64", pad_to=8)
    pfq_point, pfq_point_s = _time_call(lambda aa, bb, zz: pfq_bound(aa, bb, zz), pfq_a, pfq_b, pfq_z, iters=iters)
    pfq_mode, pfq_mode_s = _time_call(
        lambda aa, bb, zz: hypgeom_wrappers.arb_hypgeom_pfq_batch_mode_padded(
            di.interval(aa, aa),
            di.interval(bb, bb),
            di.interval(zz, zz),
            pad_to=8,
            impl="adaptive",
            prec_bits=53,
        ),
        pfq_a,
        pfq_b,
        pfq_z,
        iters=iters,
    )
    pfq_family = jnp.asarray(
        [
            di.midpoint(
                hypgeom.arb_hypgeom_pfq(
                    exact(aa),
                    exact(bb),
                    exact(zz),
                )
            )
            for aa, bb, zz in zip(pfq_a, pfq_b, pfq_z, strict=True)
        ],
        dtype=jnp.float64,
    )

    s = di.interval(jnp.asarray([1.2, 1.25], dtype=jnp.float64), jnp.asarray([1.3, 1.35], dtype=jnp.float64))
    xr = di.interval(jnp.asarray([0.3, 0.35], dtype=jnp.float64), jnp.asarray([0.35, 0.4], dtype=jnp.float64))
    lower_r, lower_s = _time_call(
        lambda ss, xx: hypgeom_wrappers.arb_hypgeom_gamma_lower_batch_mode_padded(ss, xx, pad_to=4, impl="rigorous", prec_bits=53, regularized=True),
        s,
        xr,
        iters=iters,
    )
    upper_r, upper_s = _time_call(
        lambda ss, xx: hypgeom_wrappers.arb_hypgeom_gamma_upper_batch_mode_padded(ss, xx, pad_to=4, impl="adaptive", prec_bits=53, regularized=True),
        s,
        xr,
        iters=iters,
    )
    total_r = di.fast_add(lower_r[:2], upper_r[:2])

    def _box(re_lo, re_hi, im_lo, im_hi):
        return acb_core.acb_box(
            di.interval(jnp.asarray(re_lo, dtype=jnp.float64), jnp.asarray(re_hi, dtype=jnp.float64)),
            di.interval(jnp.asarray(im_lo, dtype=jnp.float64), jnp.asarray(im_hi, dtype=jnp.float64)),
        )

    sc = _box(1.2, 1.25, -0.05, 0.05)
    zc = _box(0.3, 0.35, -0.02, 0.02)
    lower_c = hypgeom_wrappers.acb_hypgeom_gamma_lower_mode(sc, zc, impl="rigorous", prec_bits=53, regularized=True)
    upper_c = hypgeom_wrappers.acb_hypgeom_gamma_upper_mode(sc, zc, impl="adaptive", prec_bits=53, regularized=True)
    total_c = acb_core.acb_add(lower_c, upper_c)

    return {
        "onef1_point_batch_s": round(onef1_s, 6),
        "u_point_batch_s": round(u_point_s, 6),
        "u_adaptive_mode_batch_s": round(u_mode_s, 6),
        "u_point_vs_scalar_family_max_abs": round(float(jnp.max(jnp.abs(u_family - u_point))), 6),
        "u_mode_mid_vs_scalar_family_max_abs": round(float(jnp.max(jnp.abs(di.midpoint(u_mode[: u_point.shape[0]]) - u_family))), 6),
        "pfq_point_batch_s": round(pfq_point_s, 6),
        "pfq_adaptive_mode_batch_s": round(pfq_mode_s, 6),
        "pfq_point_vs_scalar_family_max_abs": round(float(jnp.max(jnp.abs(pfq_family - pfq_point))), 6),
        "pfq_mode_mid_vs_scalar_family_max_abs": round(float(jnp.max(jnp.abs(di.midpoint(pfq_mode[: pfq_point.shape[0]]) - pfq_family))), 6),
        "gamma_lower_regularized_batch_s": round(lower_s, 6),
        "gamma_upper_regularized_batch_s": round(upper_s, 6),
        "real_regularized_complement_mid_abs": round(float(jnp.max(jnp.abs(di.midpoint(total_r) - 1.0))), 6),
        "complex_regularized_complement_real_mid_abs": round(float(jnp.max(jnp.abs(acb_core.acb_midpoint(total_c).real - 1.0))), 6),
        "complex_regularized_complement_imag_mid_abs": round(float(jnp.max(jnp.abs(acb_core.acb_midpoint(total_c).imag))), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark numerically fragile special-function hardening surfaces.")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--out-json",
        default="benchmarks/results/special_function_hardening_benchmark/special_function_hardening_benchmark.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmarks/results/special_function_hardening_benchmark/special_function_hardening_benchmark.md",
    )
    args = parser.parse_args()

    payload = {
        "incomplete_bessel_i": _bench_incomplete_bessel_i(args.iters),
        "barnes_double_gamma": _bench_barnes_double_gamma(args.iters),
        "hypgeom": _bench_hypgeom(args.iters),
    }

    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "Last updated: 2026-03-26T00:00:00Z",
                "",
                "# Special-Function Hardening Benchmark",
                "",
                "Generated by `benchmarks/special_function_hardening_benchmark.py`.",
                "",
                "## Incomplete Bessel I",
                f"- quadrature s: `{payload['incomplete_bessel_i']['quadrature_s']}`",
                f"- high precision refine s: `{payload['incomplete_bessel_i']['high_precision_refine_s']}`",
                f"- auto s: `{payload['incomplete_bessel_i']['auto_s']}`",
                f"- refine vs quadrature max abs: `{payload['incomplete_bessel_i']['refine_vs_quadrature_max_abs']}`",
                f"- auto vs refine max abs: `{payload['incomplete_bessel_i']['auto_vs_refine_max_abs']}`",
                "",
                "## Barnes Double Gamma",
                f"- IFJ vector s: `{payload['barnes_double_gamma']['ifj_vector_s']}`",
                f"- provider vector s: `{payload['barnes_double_gamma']['provider_vector_s']}`",
                f"- BDG vector s: `{payload['barnes_double_gamma']['bdg_vector_s']}`",
                f"- IFJ shift recurrence abs: `{payload['barnes_double_gamma']['ifj_shift_recurrence_abs']}`",
                f"- provider vs IFJ vector max abs: `{payload['barnes_double_gamma']['provider_vs_ifj_vector_max_abs']}`",
                f"- IFJ vs BDG vector max abs: `{payload['barnes_double_gamma']['ifj_vs_bdg_vector_max_abs']}`",
                f"- IFJ diagnostics m_used: `{payload['barnes_double_gamma']['ifj_diag_m_used']}`",
                f"- IFJ diagnostics m_capped: `{payload['barnes_double_gamma']['ifj_diag_m_capped']}`",
                f"- IFJ diagnostics n_shift: `{payload['barnes_double_gamma']['ifj_diag_n_shift']}`",
                "",
                "## Hypergeom",
                f"- 1f1 point batch s: `{payload['hypgeom']['onef1_point_batch_s']}`",
                f"- U point batch s: `{payload['hypgeom']['u_point_batch_s']}`",
                f"- U adaptive mode batch s: `{payload['hypgeom']['u_adaptive_mode_batch_s']}`",
                f"- U point vs scalar family max abs: `{payload['hypgeom']['u_point_vs_scalar_family_max_abs']}`",
                f"- U mode midpoint vs scalar family max abs: `{payload['hypgeom']['u_mode_mid_vs_scalar_family_max_abs']}`",
                f"- pfq point batch s: `{payload['hypgeom']['pfq_point_batch_s']}`",
                f"- pfq adaptive mode batch s: `{payload['hypgeom']['pfq_adaptive_mode_batch_s']}`",
                f"- pfq point vs scalar family max abs: `{payload['hypgeom']['pfq_point_vs_scalar_family_max_abs']}`",
                f"- pfq mode midpoint vs scalar family max abs: `{payload['hypgeom']['pfq_mode_mid_vs_scalar_family_max_abs']}`",
                f"- regularized gamma lower batch s: `{payload['hypgeom']['gamma_lower_regularized_batch_s']}`",
                f"- regularized gamma upper batch s: `{payload['hypgeom']['gamma_upper_regularized_batch_s']}`",
                f"- real complement mid abs: `{payload['hypgeom']['real_regularized_complement_mid_abs']}`",
                f"- complex complement real mid abs: `{payload['hypgeom']['complex_regularized_complement_real_mid_abs']}`",
                f"- complex complement imag mid abs: `{payload['hypgeom']['complex_regularized_complement_imag_mid_abs']}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
