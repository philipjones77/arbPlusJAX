from __future__ import annotations

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


import argparse
import platform
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import nufft

try:
    import nufftax  # type: ignore

    HAS_NUFFTAX = True
except Exception:
    HAS_NUFFTAX = False

try:
    import jax_finufft  # type: ignore

    HAS_JAX_FINUFFT = True
except Exception:
    HAS_JAX_FINUFFT = False


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


def _case(n_points: int, n_modes: int, seed: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    points = jnp.asarray(rng.random(n_points), dtype=jnp.float64)
    values = jnp.asarray(rng.normal(size=n_points) + 1j * rng.normal(size=n_points), dtype=jnp.complex128)
    modes = jnp.asarray(rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes), dtype=jnp.complex128)
    return points, values, modes


def _try_nufftax_type1(points: jax.Array, values: jax.Array, n_modes: int):
    for name in ("nufft1", "type1", "nufft_type1"):
        fn = getattr(nufftax, name, None)
        if fn is not None:
            return fn(points, values, n_modes)
    raise AttributeError("nufftax type-1 entrypoint not found")


def _try_nufftax_type2(points: jax.Array, modes: jax.Array):
    for name in ("nufft2", "type2", "nufft_type2"):
        fn = getattr(nufftax, name, None)
        if fn is not None:
            return fn(points, modes)
    raise AttributeError("nufftax type-2 entrypoint not found")


def _try_jax_finufft_type1(points: jax.Array, values: jax.Array, n_modes: int):
    for name in ("nufft1", "type1", "nufft_type1"):
        fn = getattr(jax_finufft, name, None)
        if fn is not None:
            return fn(points, values, n_modes)
    raise AttributeError("jax_finufft type-1 entrypoint not found")


def _try_jax_finufft_type2(points: jax.Array, modes: jax.Array):
    for name in ("nufft2", "type2", "nufft_type2"):
        fn = getattr(jax_finufft, name, None)
        if fn is not None:
            return fn(points, modes)
    raise AttributeError("jax_finufft type-2 entrypoint not found")


def run_backend_suite(n_points: int, n_modes: int, warmup: int, runs: int) -> dict[str, float]:
    points, values, modes = _case(n_points, n_modes, seed=20260321)
    internal_type1 = jax.jit(lambda p, v: nufft.nufft_type1(p, v, n_modes, method="lanczos"))
    internal_type2 = jax.jit(lambda p, m: nufft.nufft_type2(p, m, method="lanczos"))
    direct_type1 = jax.jit(lambda p, v: nufft.nufft_type1_direct(p, v, n_modes))
    direct_type2 = jax.jit(lambda p, m: nufft.nufft_type2_direct(p, m))

    results = {
        "internal_nufft_type1_lanczos_s": _time_call(internal_type1, points, values, warmup=warmup, runs=runs),
        "internal_nufft_type2_lanczos_s": _time_call(internal_type2, points, modes, warmup=warmup, runs=runs),
        "internal_nufft_type1_direct_s": _time_call(direct_type1, points, values, warmup=warmup, runs=runs),
        "internal_nufft_type2_direct_s": _time_call(direct_type2, points, modes, warmup=warmup, runs=runs),
    }

    ref_type1 = internal_type1(points, values)
    ref_type2 = internal_type2(points, modes)

    if HAS_NUFFTAX:
        try:
            results["nufftax_type1_s"] = _time_call(
                _try_nufftax_type1, points, values, n_modes, warmup=warmup, runs=runs
            )
            got1 = _try_nufftax_type1(points, values, n_modes)
            results["nufftax_type1_relerr"] = float(
                jnp.linalg.norm(got1 - ref_type1) / jnp.maximum(jnp.linalg.norm(ref_type1), 1e-12)
            )
        except Exception:
            results["nufftax_type1_available"] = 0.0
        try:
            results["nufftax_type2_s"] = _time_call(_try_nufftax_type2, points, modes, warmup=warmup, runs=runs)
            got2 = _try_nufftax_type2(points, modes)
            results["nufftax_type2_relerr"] = float(
                jnp.linalg.norm(got2 - ref_type2) / jnp.maximum(jnp.linalg.norm(ref_type2), 1e-12)
            )
        except Exception:
            results["nufftax_type2_available"] = 0.0
    else:
        results["nufftax_available"] = 0.0

    if HAS_JAX_FINUFFT:
        try:
            results["jax_finufft_type1_s"] = _time_call(
                _try_jax_finufft_type1, points, values, n_modes, warmup=warmup, runs=runs
            )
            got1 = _try_jax_finufft_type1(points, values, n_modes)
            results["jax_finufft_type1_relerr"] = float(
                jnp.linalg.norm(got1 - ref_type1) / jnp.maximum(jnp.linalg.norm(ref_type1), 1e-12)
            )
        except Exception:
            results["jax_finufft_type1_available"] = 0.0
        try:
            results["jax_finufft_type2_s"] = _time_call(
                _try_jax_finufft_type2, points, modes, warmup=warmup, runs=runs
            )
            got2 = _try_jax_finufft_type2(points, modes)
            results["jax_finufft_type2_relerr"] = float(
                jnp.linalg.norm(got2 - ref_type2) / jnp.maximum(jnp.linalg.norm(ref_type2), 1e-12)
            )
        except Exception:
            results["jax_finufft_type2_available"] = 0.0
    else:
        results["jax_finufft_available"] = 0.0

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark internal and optional external JAX NUFFT backends.")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--n-modes", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"jax_backend: {jax.default_backend()}")
    print(f"n_points: {args.n_points}")
    print(f"n_modes: {args.n_modes}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"has_nufftax: {HAS_NUFFTAX}")
    print(f"has_jax_finufft: {HAS_JAX_FINUFFT}")

    stats = run_backend_suite(args.n_points, args.n_modes, args.warmup, args.runs)
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
