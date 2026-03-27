from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


acb_core = None
barnesg = None
double_gamma = None
di = None


def _load_barnes_modules():
    global acb_core, barnesg, double_gamma, di
    if acb_core is None:
        from arbplusjax import acb_core as _acb_core
        from arbplusjax import barnesg as _barnesg
        from arbplusjax import double_gamma as _double_gamma
        from arbplusjax import double_interval as _di

        acb_core = _acb_core
        barnesg = _barnesg
        double_gamma = _double_gamma
        di = _di


def _time_call(fn, *args, iters: int = 1) -> float:
    started = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - started) / float(iters)


def _legacy_vector(zs: jax.Array, tau: float) -> jax.Array:
    _load_barnes_modules()
    return jax.vmap(lambda zz: double_gamma.bdg_barnesdoublegamma(zz, tau, prec_bits=80))(zs)


def _ifj_vector(zs: jax.Array, tau: float) -> jax.Array:
    _load_barnes_modules()
    return double_gamma.ifj_barnesdoublegamma(zs, tau, dps=60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Barnes G and double-gamma production surfaces.")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--prec-bits", type=int, default=80)
    parser.add_argument("--max-m-cap", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/special/benchmark_barnes_double_gamma.json"),
    )
    args = parser.parse_args()
    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64
    effective_iters = min(args.iters, 1) if args.smoke else args.iters
    _load_barnes_modules()

    z_scalar = jnp.asarray(1.7 + 0.1j, dtype=complex_dtype)
    tau = 0.5
    zs = jnp.asarray(
        [
            1.2 + 0.1j,
            1.5 + 0.15j,
            1.8 + 0.2j,
        ],
        dtype=complex_dtype,
    )
    anchors = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=real_dtype)
    expected = jnp.asarray([1.0, 1.0, 1.0, 2.0], dtype=complex_dtype)

    legacy_scalar_s = _time_call(lambda z: double_gamma.bdg_barnesdoublegamma(z, tau, prec_bits=args.prec_bits), z_scalar, iters=effective_iters)
    ifj_scalar_s = _time_call(lambda z: double_gamma.ifj_barnesdoublegamma(z, tau, dps=args.dps), z_scalar, iters=effective_iters)
    legacy_vector_s = _time_call(_legacy_vector, zs, tau, iters=effective_iters)
    ifj_vector_s = _time_call(lambda arr, t: double_gamma.ifj_barnesdoublegamma(arr, t, dps=args.dps), zs, tau, iters=effective_iters)

    legacy_shift = double_gamma.bdg_barnesdoublegamma(z_scalar + 1.0, tau, prec_bits=args.prec_bits) / double_gamma.bdg_barnesdoublegamma(z_scalar, tau, prec_bits=args.prec_bits)
    ifj_shift = double_gamma.ifj_barnesdoublegamma(z_scalar + 1.0, tau, dps=args.dps) / double_gamma.ifj_barnesdoublegamma(z_scalar, tau, dps=args.dps)
    shift_target = jnp.exp(barnesg._complex_loggamma(z_scalar / tau))
    legacy_shift_err = jnp.abs(legacy_shift - shift_target)
    ifj_shift_err = jnp.abs(ifj_shift - shift_target)

    legacy_anchor = jnp.asarray([double_gamma.bdg_barnesdoublegamma(x, 1.0, prec_bits=args.prec_bits) for x in anchors], dtype=complex_dtype)
    ifj_anchor = jnp.asarray([double_gamma.ifj_barnesdoublegamma(x, 1.0, dps=args.dps) for x in anchors], dtype=complex_dtype)
    legacy_anchor_err = jnp.max(jnp.abs(legacy_anchor - expected))
    ifj_anchor_err = jnp.max(jnp.abs(ifj_anchor - expected))

    diagnostics = double_gamma.ifj_barnesdoublegamma_diagnostics(0.2 + 0.05j, 1.0, dps=args.dps, max_m_cap=args.max_m_cap)

    point = acb_core.acb_box(di.interval(1.3, 1.3), di.interval(0.2, 0.2))
    g_box = acb_core.acb_barnes_g(point)
    log_box = acb_core.acb_log_barnes_g(point)
    alias_consistency = jnp.abs(jnp.exp(acb_core.acb_midpoint(log_box)) - acb_core.acb_midpoint(g_box))

    rows = {
        "legacy_scalar_s": float(legacy_scalar_s),
        "ifj_scalar_s": float(ifj_scalar_s),
        "legacy_vector_s": float(legacy_vector_s),
        "ifj_vector_s": float(ifj_vector_s),
        "legacy_shift_err_abs": float(legacy_shift_err),
        "ifj_shift_err_abs": float(ifj_shift_err),
        "legacy_tau1_anchor_max_abs_err": float(legacy_anchor_err),
        "ifj_tau1_anchor_max_abs_err": float(ifj_anchor_err),
        "ifj_diag_m_base": float(diagnostics.m_base),
        "ifj_diag_m_used": float(diagnostics.m_used),
        "ifj_diag_n_shift": float(diagnostics.n_shift),
        "ifj_diag_m_capped": float(int(diagnostics.m_capped)),
        "acb_barnes_g_alias_consistency_abs": float(alias_consistency),
    }

    records = [
        BenchmarkRecord(
            benchmark_name="benchmark_barnes_double_gamma.py",
            concern="special_speed",
            category="special",
            implementation="legacy_bdg",
            operation="barnes_double_gamma_scalar",
            device=jax.default_backend(),
            dtype="complex128" if args.dtype == "float64" else "complex64",
            warm_time_s=float(legacy_scalar_s),
            accuracy_abs=float(legacy_shift_err),
            measurements=(
                BenchmarkMeasurement(name="iters", value=effective_iters, unit="calls"),
                BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                BenchmarkMeasurement(name="tau1_anchor_max_abs_err", value=float(legacy_anchor_err)),
            ),
            notes="Legacy Barnes/double-gamma scalar path.",
        ),
        BenchmarkRecord(
            benchmark_name="benchmark_barnes_double_gamma.py",
            concern="special_speed",
            category="special",
            implementation="ifj",
            operation="barnes_double_gamma_scalar",
            device=jax.default_backend(),
            dtype="complex128" if args.dtype == "float64" else "complex64",
            warm_time_s=float(ifj_scalar_s),
            accuracy_abs=float(ifj_shift_err),
            measurements=(
                BenchmarkMeasurement(name="iters", value=effective_iters, unit="calls"),
                BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                BenchmarkMeasurement(name="tau1_anchor_max_abs_err", value=float(ifj_anchor_err)),
                BenchmarkMeasurement(name="m_used", value=float(diagnostics.m_used)),
                BenchmarkMeasurement(name="n_shift", value=float(diagnostics.n_shift)),
                BenchmarkMeasurement(name="m_capped", value=float(int(diagnostics.m_capped))),
                BenchmarkMeasurement(name="alias_consistency_abs", value=float(alias_consistency)),
            ),
            notes="IFJ Barnes/double-gamma scalar path with diagnostics-backed production surface.",
        ),
    ]
    if not args.smoke:
        records.extend(
            (
                BenchmarkRecord(
                    benchmark_name="benchmark_barnes_double_gamma.py",
                    concern="special_speed",
                    category="special",
                    implementation="legacy_bdg",
                    operation="barnes_double_gamma_vector",
                    device=jax.default_backend(),
                    dtype="complex128" if args.dtype == "float64" else "complex64",
                    warm_time_s=float(legacy_vector_s),
                    measurements=(BenchmarkMeasurement(name="requested_dtype", value=args.dtype),),
                    notes="Legacy vectorized Barnes/double-gamma path.",
                ),
                BenchmarkRecord(
                    benchmark_name="benchmark_barnes_double_gamma.py",
                    concern="special_speed",
                    category="special",
                    implementation="ifj",
                    operation="barnes_double_gamma_vector",
                    device=jax.default_backend(),
                    dtype="complex128" if args.dtype == "float64" else "complex64",
                    warm_time_s=float(ifj_vector_s),
                    measurements=(
                        BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                        BenchmarkMeasurement(name="m_used", value=float(diagnostics.m_used)),
                    ),
                    notes="IFJ vectorized Barnes/double-gamma path.",
                ),
            )
        )
    report = BenchmarkReport(
        benchmark_name="benchmark_barnes_double_gamma.py",
        concern="special_speed",
        category="special",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode=args.jax_mode,
        ),
        notes="Barnes G and double-gamma benchmark. Stdout preserves metric-style lines for notebook compatibility.",
    )
    write_benchmark_report(args.output, report)

    for key, value in rows.items():
        if key in {"ifj_diag_m_capped"}:
            print(f"{key}: {int(value)}")
        elif key.startswith("ifj_diag_"):
            print(f"{key}: {value:.0f}")
        elif key.endswith("_s"):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
