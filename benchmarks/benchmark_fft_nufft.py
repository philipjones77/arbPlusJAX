from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from arbplusjax import dft
from arbplusjax import nufft


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
    rng = np.random.default_rng(20260317)

    x_pow2 = jnp.asarray(rng.normal(size=1024) + 1j * rng.normal(size=1024), dtype=jnp.complex128)
    x_prime = jnp.asarray(rng.normal(size=257) + 1j * rng.normal(size=257), dtype=jnp.complex128)
    x2 = jnp.asarray(rng.normal(size=(64, 48)) + 1j * rng.normal(size=(64, 48)), dtype=jnp.complex128)
    x3 = jnp.asarray(rng.normal(size=(16, 12, 10)) + 1j * rng.normal(size=(16, 12, 10)), dtype=jnp.complex128)
    xb_prime = jnp.stack([jnp.real(x_prime), jnp.real(x_prime), jnp.imag(x_prime), jnp.imag(x_prime)], axis=-1)
    dft_precomp = dft.make_dft_precomp(257)
    dft_plan_point = dft.dft_matvec_cached_prepare_point(257)
    dft_plan_basic = dft.dft_matvec_cached_prepare_basic(257)
    x_prime_batch = jnp.stack([x_prime, (0.5 - 0.25j) * x_prime], axis=0)
    xb_prime_batch = jnp.stack([xb_prime, xb_prime], axis=0)

    points_small = jnp.asarray(rng.random(256), dtype=jnp.float64)
    values_small = jnp.asarray(rng.normal(size=256) + 1j * rng.normal(size=256), dtype=jnp.complex128)
    points_large = jnp.asarray(rng.random(2048), dtype=jnp.float64)
    values_large = jnp.asarray(rng.normal(size=2048) + 1j * rng.normal(size=2048), dtype=jnp.complex128)
    values_large_batch = jnp.stack([values_large, (1.0 + 0.1j) * values_large], axis=0)
    modes_large = jnp.asarray(rng.normal(size=512) + 1j * rng.normal(size=512), dtype=jnp.complex128)
    nufft_plan_type1 = nufft.nufft_type1_cached_prepare(points_large, 512, method="lanczos")
    nufft_plan_type2 = nufft.nufft_type2_cached_prepare(points_large, 512, method="lanczos")
    points_2d = jnp.asarray(rng.random((384, 2)), dtype=jnp.float64)
    values_2d = jnp.asarray(rng.normal(size=384) + 1j * rng.normal(size=384), dtype=jnp.complex128)
    modes_2d = jnp.asarray(rng.normal(size=(48, 40)) + 1j * rng.normal(size=(48, 40)), dtype=jnp.complex128)
    points_3d = jnp.asarray(rng.random((160, 3)), dtype=jnp.float64)
    values_3d = jnp.asarray(rng.normal(size=160) + 1j * rng.normal(size=160), dtype=jnp.complex128)
    modes_3d = jnp.asarray(rng.normal(size=(12, 10, 8)) + 1j * rng.normal(size=(12, 10, 8)), dtype=jnp.complex128)

    cases = [
        ("dft_power2_s", dft.dft_jit, x_pow2),
        ("dft_prime_bluestein_s", dft.dft_jit, x_prime),
        ("dft_prime_precomp_s", lambda x: dft.dft_bluestein_precomp(x, precomp=dft_precomp), x_prime),
        ("dft_prime_cached_point_s", dft.dft_matvec_cached_apply_point_jit, dft_plan_point, x_prime),
        ("dft_prime_cached_basic_s", dft.dft_matvec_cached_apply_basic_jit, dft_plan_basic, xb_prime),
        ("dft_prime_batch_point_s", dft.dft_matvec_batch_fixed_point_jit, x_prime_batch),
        ("dft_prime_cached_batch_point_s", dft.dft_matvec_cached_apply_batch_fixed_point_jit, dft_plan_point, x_prime_batch),
        ("dft_prime_batch_basic_s", dft.dft_matvec_batch_fixed_basic_jit, xb_prime_batch),
        ("dft_prime_cached_batch_basic_s", dft.dft_matvec_cached_apply_batch_fixed_basic_jit, dft_plan_basic, xb_prime_batch),
        ("dft2_s", dft.dft2_jit, x2),
        ("dft3_s", dft.dft3_jit, x3),
        ("acb_dft_prime_point_s", dft.acb_dft_jit, xb_prime),
        ("nufft_type1_direct_s", lambda p, v: nufft.nufft_type1(p, v, 128, method="direct"), points_small, values_small),
        ("nufft_type1_lanczos_s", lambda p, v: nufft.nufft_type1(p, v, 512, method="lanczos"), points_large, values_large),
        ("nufft_type2_lanczos_s", lambda p, m: nufft.nufft_type2(p, m, method="lanczos"), points_large, modes_large),
        ("nufft_type1_cached_lanczos_s", nufft.nufft_type1_cached_apply_jit, nufft_plan_type1, values_large),
        ("nufft_type2_cached_lanczos_s", nufft.nufft_type2_cached_apply_jit, nufft_plan_type2, modes_large),
        ("nufft_type1_cached_batch_s", nufft.nufft_type1_cached_apply_batch_fixed_jit, nufft_plan_type1, values_large_batch),
        ("nufft_type1_2d_lanczos_s", lambda p, v: nufft.nufft_type1_2d(p, v, (48, 40), method="lanczos"), points_2d, values_2d),
        ("nufft_type2_2d_lanczos_s", lambda p, m: nufft.nufft_type2_2d(p, m, method="lanczos"), points_2d, modes_2d),
        ("nufft_type1_3d_lanczos_s", lambda p, v: nufft.nufft_type1_3d(p, v, (12, 10, 8), method="lanczos"), points_3d, values_3d),
        ("nufft_type2_3d_lanczos_s", lambda p, m: nufft.nufft_type2_3d(p, m, method="lanczos"), points_3d, modes_3d),
    ]

    print("name,time_s")
    for case in cases:
        name, fn, *args = case
        bench_name, seconds = _bench(name, fn, *args)
        print(f"{bench_name},{seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
