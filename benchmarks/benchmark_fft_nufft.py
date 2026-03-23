from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import dft
from arbplusjax import nufft
from benchmarks.schema import BenchmarkMeasurement
from benchmarks.schema import BenchmarkRecord
from benchmarks.schema import BenchmarkReport
from benchmarks.schema import write_benchmark_report
from tools.runtime_manifest import collect_runtime_manifest


def _block(value):
    if isinstance(value, tuple):
        for item in value:
            _block(item)
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _bench(name: str, fn, *args, repeat: int = 5) -> tuple[str, float]:
    _block(fn(*args))
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn(*args)
        _block(out)
        times.append(time.perf_counter() - start)
    return name, min(times)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FFT and NUFFT transform surfaces in pure JAX.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/benchmarks/outputs/transform/benchmark_fft_nufft.json"),
    )
    args = parser.parse_args()

    real_dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    complex_dtype = jnp.complex128 if args.dtype == "float64" else jnp.complex64

    rng = np.random.default_rng(20260317)

    x_pow2 = jnp.asarray(rng.normal(size=1024) + 1j * rng.normal(size=1024), dtype=complex_dtype)
    x_prime = jnp.asarray(rng.normal(size=257) + 1j * rng.normal(size=257), dtype=complex_dtype)
    x2 = jnp.asarray(rng.normal(size=(64, 48)) + 1j * rng.normal(size=(64, 48)), dtype=complex_dtype)
    x3 = jnp.asarray(rng.normal(size=(16, 12, 10)) + 1j * rng.normal(size=(16, 12, 10)), dtype=complex_dtype)
    xb_prime = jnp.stack([jnp.real(x_prime), jnp.real(x_prime), jnp.imag(x_prime), jnp.imag(x_prime)], axis=-1)
    dft_precomp = dft.make_dft_precomp(257)
    dft_plan_point = dft.dft_matvec_cached_prepare_point(257)
    dft_plan_basic = dft.dft_matvec_cached_prepare_basic(257)
    x_prime_batch = jnp.stack([x_prime, (0.5 - 0.25j) * x_prime], axis=0)
    xb_prime_batch = jnp.stack([xb_prime, xb_prime], axis=0)

    points_small = jnp.asarray(rng.random(256), dtype=real_dtype)
    values_small = jnp.asarray(rng.normal(size=256) + 1j * rng.normal(size=256), dtype=complex_dtype)
    points_large = jnp.asarray(rng.random(2048), dtype=real_dtype)
    values_large = jnp.asarray(rng.normal(size=2048) + 1j * rng.normal(size=2048), dtype=complex_dtype)
    values_large_batch = jnp.stack([values_large, (1.0 + 0.1j) * values_large], axis=0)
    modes_large = jnp.asarray(rng.normal(size=512) + 1j * rng.normal(size=512), dtype=complex_dtype)
    nufft_plan_type1 = nufft.nufft_type1_cached_prepare(points_large, 512, method="lanczos")
    nufft_plan_type2 = nufft.nufft_type2_cached_prepare(points_large, 512, method="lanczos")
    points_2d = jnp.asarray(rng.random((384, 2)), dtype=real_dtype)
    values_2d = jnp.asarray(rng.normal(size=384) + 1j * rng.normal(size=384), dtype=complex_dtype)
    modes_2d = jnp.asarray(rng.normal(size=(48, 40)) + 1j * rng.normal(size=(48, 40)), dtype=complex_dtype)
    points_3d = jnp.asarray(rng.random((160, 3)), dtype=real_dtype)
    values_3d = jnp.asarray(rng.normal(size=160) + 1j * rng.normal(size=160), dtype=complex_dtype)
    modes_3d = jnp.asarray(rng.normal(size=(12, 10, 8)) + 1j * rng.normal(size=(12, 10, 8)), dtype=complex_dtype)

    cases = [
        ("dft_power2_s", "repo_native", "dft_power2", dft.dft_jit, x_pow2),
        ("dft_prime_bluestein_s", "repo_native", "dft_prime_bluestein", dft.dft_jit, x_prime),
        ("dft_prime_precomp_s", "cached", "dft_prime_precomp", lambda x: dft.dft_bluestein_precomp(x, precomp=dft_precomp), x_prime),
        ("dft_prime_cached_point_s", "cached", "dft_prime_cached_point", dft.dft_matvec_cached_apply_point_jit, dft_plan_point, x_prime),
        ("dft_prime_cached_basic_s", "cached", "dft_prime_cached_basic", dft.dft_matvec_cached_apply_basic_jit, dft_plan_basic, xb_prime),
        ("dft_prime_batch_point_s", "batched", "dft_prime_batch_point", dft.dft_matvec_batch_fixed_point_jit, x_prime_batch),
        ("dft_prime_cached_batch_point_s", "cached_batched", "dft_prime_cached_batch_point", dft.dft_matvec_cached_apply_batch_fixed_point_jit, dft_plan_point, x_prime_batch),
        ("dft_prime_batch_basic_s", "batched", "dft_prime_batch_basic", dft.dft_matvec_batch_fixed_basic_jit, xb_prime_batch),
        ("dft_prime_cached_batch_basic_s", "cached_batched", "dft_prime_cached_batch_basic", dft.dft_matvec_cached_apply_batch_fixed_basic_jit, dft_plan_basic, xb_prime_batch),
        ("dft2_s", "repo_native", "dft2", dft.dft2_jit, x2),
        ("dft3_s", "repo_native", "dft3", dft.dft3_jit, x3),
        ("acb_dft_prime_point_s", "repo_native", "acb_dft_prime", dft.acb_dft_jit, xb_prime),
        ("nufft_type1_direct_s", "direct", "nufft_type1_direct", lambda p, v: nufft.nufft_type1(p, v, 128, method="direct"), points_small, values_small),
        ("nufft_type1_lanczos_s", "repo_native", "nufft_type1_lanczos", lambda p, v: nufft.nufft_type1(p, v, 512, method="lanczos"), points_large, values_large),
        ("nufft_type2_lanczos_s", "repo_native", "nufft_type2_lanczos", lambda p, m: nufft.nufft_type2(p, m, method="lanczos"), points_large, modes_large),
        ("nufft_type1_cached_lanczos_s", "cached", "nufft_type1_cached_lanczos", nufft.nufft_type1_cached_apply_jit, nufft_plan_type1, values_large),
        ("nufft_type2_cached_lanczos_s", "cached", "nufft_type2_cached_lanczos", nufft.nufft_type2_cached_apply_jit, nufft_plan_type2, modes_large),
        ("nufft_type1_cached_batch_s", "cached_batched", "nufft_type1_cached_batch", nufft.nufft_type1_cached_apply_batch_fixed_jit, nufft_plan_type1, values_large_batch),
        ("nufft_type1_2d_lanczos_s", "repo_native", "nufft_type1_2d_lanczos", lambda p, v: nufft.nufft_type1_2d(p, v, (48, 40), method="lanczos"), points_2d, values_2d),
        ("nufft_type2_2d_lanczos_s", "repo_native", "nufft_type2_2d_lanczos", lambda p, m: nufft.nufft_type2_2d(p, m, method="lanczos"), points_2d, modes_2d),
        ("nufft_type1_3d_lanczos_s", "repo_native", "nufft_type1_3d_lanczos", lambda p, v: nufft.nufft_type1_3d(p, v, (12, 10, 8), method="lanczos"), points_3d, values_3d),
        ("nufft_type2_3d_lanczos_s", "repo_native", "nufft_type2_3d_lanczos", lambda p, m: nufft.nufft_type2_3d(p, m, method="lanczos"), points_3d, modes_3d),
    ]

    records: list[BenchmarkRecord] = []
    csv_rows = [("name", "time_s")]
    for csv_name, implementation, operation, fn, *fn_args in cases:
        _, seconds = _bench(csv_name, fn, *fn_args, repeat=args.repeat)
        csv_rows.append((csv_name, f"{seconds:.6f}"))
        arr = fn_args[-1]
        records.append(
            BenchmarkRecord(
                benchmark_name="benchmark_fft_nufft.py",
                concern="transform_speed",
                category="transform",
                implementation=implementation,
                operation=operation,
                device=jax.default_backend(),
                dtype=str(jnp.asarray(arr).dtype),
                warm_time_s=seconds,
                measurements=(
                    BenchmarkMeasurement(name="repeat", value=args.repeat, unit="calls"),
                    BenchmarkMeasurement(name="requested_dtype", value=args.dtype),
                ),
                notes="Legacy transform benchmark normalized onto the shared benchmark schema; stdout remains CSV for notebook compatibility.",
            )
        )

    report = BenchmarkReport(
        benchmark_name="benchmark_fft_nufft.py",
        concern="transform_speed",
        category="transform",
        records=tuple(records),
        environment=collect_runtime_manifest(
            Path(__file__).resolve().parents[1],
            jax_mode="gpu" if jax.default_backend() in {"gpu", "cuda"} else "cpu",
        ),
        notes="FFT/NUFFT transform benchmark. Stdout preserves CSV compatibility for the canonical transform notebook.",
    )
    write_benchmark_report(args.output, report)

    for name, seconds in csv_rows:
        print(f"{name},{seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
