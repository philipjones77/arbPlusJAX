from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import hypgeom
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _timer(fn, *, iters: int = 50) -> tuple[float, float]:
    out = fn()
    jax.block_until_ready(out)
    start = time.perf_counter()
    cold = fn()
    jax.block_until_ready(cold)
    cold_s = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        jax.block_until_ready(out)
    warm_s = (time.perf_counter() - start) / iters
    return cold_s, warm_s


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark extra hypergeometric JAX kernels.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced subset for pytest-owned schema checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/hypgeom/benchmark_hypgeom_extra.json"),
    )
    args = parser.parse_args()

    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    x = jnp.array([0.2, 0.3], dtype=real_dtype)
    s = jnp.array([1.0, 1.0], dtype=real_dtype)
    a = jnp.array([0.5, 0.5], dtype=real_dtype)
    b = jnp.array([1.5, 1.5], dtype=real_dtype)

    bench = [
        ("fresnel", lambda: hypgeom.arb_hypgeom_fresnel_batch_jit(jnp.stack([x, x]))),
        ("ei", lambda: hypgeom.arb_hypgeom_ei_batch_jit(jnp.stack([x, x]))),
        ("si", lambda: hypgeom.arb_hypgeom_si_batch_jit(jnp.stack([x, x]))),
        ("ci", lambda: hypgeom.arb_hypgeom_ci_batch_jit(jnp.stack([x, x]))),
        ("shi", lambda: hypgeom.arb_hypgeom_shi_batch_jit(jnp.stack([x, x]))),
        ("chi", lambda: hypgeom.arb_hypgeom_chi_batch_jit(jnp.stack([x, x]))),
        ("li", lambda: hypgeom.arb_hypgeom_li_batch_jit(jnp.stack([x, x]), offset=1)),
        ("dilog", lambda: hypgeom.arb_hypgeom_dilog_batch_jit(jnp.stack([x, x]))),
        ("airy", lambda: hypgeom.arb_hypgeom_airy_batch_jit(jnp.stack([x, x]))),
        ("expint", lambda: hypgeom.arb_hypgeom_expint_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("gamma_lower", lambda: hypgeom.arb_hypgeom_gamma_lower_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("gamma_upper", lambda: hypgeom.arb_hypgeom_gamma_upper_batch_jit(jnp.stack([s, s]), jnp.stack([x, x]))),
        ("beta_lower", lambda: hypgeom.arb_hypgeom_beta_lower_batch_jit(jnp.stack([a, a]), jnp.stack([b, b]), jnp.stack([x, x]))),
        ("cheb_t", lambda: hypgeom.arb_hypgeom_chebyshev_t_batch_jit(args.n, jnp.stack([x, x]))),
        ("cheb_u", lambda: hypgeom.arb_hypgeom_chebyshev_u_batch_jit(args.n, jnp.stack([x, x]))),
        ("laguerre", lambda: hypgeom.arb_hypgeom_laguerre_l_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("hermite", lambda: hypgeom.arb_hypgeom_hermite_h_batch_jit(args.n, jnp.stack([x, x]))),
        ("legendre_p", lambda: hypgeom.arb_hypgeom_legendre_p_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("legendre_q", lambda: hypgeom.arb_hypgeom_legendre_q_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("jacobi", lambda: hypgeom.arb_hypgeom_jacobi_p_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([b, b]), jnp.stack([x, x]))),
        ("gegenbauer", lambda: hypgeom.arb_hypgeom_gegenbauer_c_batch_jit(args.n, jnp.stack([a, a]), jnp.stack([x, x]))),
        ("central_bin", lambda: hypgeom.arb_hypgeom_central_bin_ui_batch_jit(jnp.array([args.n, args.n + 1]))),
    ]
    if args.smoke:
        bench = bench[:6]

    effective_iters = min(args.iters, 5) if args.smoke else args.iters
    records = []
    for name, fn in bench:
        cold_s, warm_s = _timer(fn, iters=effective_iters)
        print(f"{name:14s} {warm_s*1e6:8.2f} us")
        records.append(
            BenchmarkRecord(
                benchmark_name="benchmark_hypgeom_extra.py",
                concern="scalar_speed",
                category="special",
                implementation="hypgeom",
                operation=name,
                device=jax.default_backend(),
                dtype=args.dtype,
                cold_time_s=float(cold_s),
                warm_time_s=float(warm_s),
                measurements=(
                    BenchmarkMeasurement(name="iters", value=effective_iters),
                    BenchmarkMeasurement(name="n", value=args.n),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                    BenchmarkMeasurement(name="smoke", value=args.smoke),
                ),
                notes="Legacy hypergeometric extra benchmark normalized onto the shared benchmark schema; stdout remains summary-style for notebook compatibility.",
            )
        )

    report = BenchmarkReport(
        benchmark_name="benchmark_hypgeom_extra.py",
        concern="scalar_speed",
        category="special",
        records=tuple(records),
        environment=collect_runtime_manifest(Path(__file__).resolve().parents[1], jax_mode=args.jax_mode),
        notes="Hypergeometric extras benchmark with explicit dtype and requested runtime mode controls.",
    )
    write_benchmark_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
