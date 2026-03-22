from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import api, boost_hypgeom, point_wrappers


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "reports" / "hypgeom_point_kernel_benchmark.md"


def _timeit(fn, *args, repeats: int = 5, **kwargs) -> float:
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    return (time.perf_counter() - start) / repeats * 1e3


def _real_batch(lo: float, hi: float, n: int) -> jax.Array:
    return jnp.linspace(jnp.float32(lo), jnp.float32(hi), n)


def _pfq_real_params(n: int, p: int, start: float) -> jax.Array:
    base = jnp.linspace(jnp.float32(start), jnp.float32(start + 0.2), n)
    cols = [base + jnp.float32(0.1 * i) for i in range(p)]
    return jnp.stack(cols, axis=1)


def main() -> None:
    n = 128
    pad_to = 128
    a = _real_batch(1.1, 1.4, n)
    b = _real_batch(2.1, 2.4, n)
    c = _real_batch(2.8, 3.1, n)
    z_small = _real_batch(0.1, 0.2, n)
    z_mid = _real_batch(0.1, 0.4, n)
    z_u = _real_batch(0.6, 1.2, n)
    pfq_a = _pfq_real_params(n, 2, 0.6)
    pfq_b = _pfq_real_params(n, 1, 1.4)
    a_iv = jnp.stack((a, a), axis=-1)
    b_iv = jnp.stack((b, b), axis=-1)
    c_iv = jnp.stack((c, c), axis=-1)
    z_small_iv = jnp.stack((z_small, z_small), axis=-1)
    z_mid_iv = jnp.stack((z_mid, z_mid), axis=-1)
    z_u_iv = jnp.stack((z_u, z_u), axis=-1)
    pfq_a_iv = jnp.stack((pfq_a, pfq_a), axis=-1)
    pfq_b_iv = jnp.stack((pfq_b, pfq_b), axis=-1)

    rows = [
        (
            "hypgeom.arb_hypgeom_1f1",
            lambda: api.eval_point_batch("hypgeom.arb_hypgeom_1f1", a, b, z_mid, dtype="float32"),
            lambda: point_wrappers.arb_hypgeom_1f1_batch_fixed_point(a, b, z_mid),
        ),
        (
            "hypgeom.arb_hypgeom_2f1",
            lambda: api.eval_point_batch("hypgeom.arb_hypgeom_2f1", a, b, c, z_small, dtype="float32"),
            lambda: point_wrappers.arb_hypgeom_2f1_batch_fixed_point(a, b, c, z_small),
        ),
        (
            "hypgeom.arb_hypgeom_u",
            lambda: api.eval_point_batch("hypgeom.arb_hypgeom_u", a, b, z_u, dtype="float32"),
            lambda: point_wrappers.arb_hypgeom_u_batch_fixed_point(a, b, z_u),
        ),
        (
            "hypgeom.arb_hypgeom_pfq",
            lambda: api.eval_point_batch("hypgeom.arb_hypgeom_pfq", pfq_a, pfq_b, z_mid, dtype="float32"),
            lambda: point_wrappers.arb_hypgeom_pfq_batch_fixed_point(pfq_a, pfq_b, z_mid),
        ),
        (
            "boost_hypergeometric_1f1",
            lambda: api.eval_point_batch("boost_hypergeometric_1f1", a, b, z_mid, dtype="float32", pad_to=pad_to),
            lambda: boost_hypgeom.boost_hypergeometric_1f1_batch_fixed_point(a, b, z_mid),
        ),
        (
            "boost_hyp2f1_series",
            lambda: api.eval_point_batch("boost_hyp2f1_series", a, b, c, z_small, dtype="float32", pad_to=pad_to),
            lambda: boost_hypgeom.boost_hyp2f1_series_batch_fixed_point(a, b, c, z_small),
        ),
        (
            "boost_hypergeometric_pfq",
            lambda: api.eval_point_batch("boost_hypergeometric_pfq", pfq_a, pfq_b, z_mid, dtype="float32", pad_to=pad_to),
            lambda: boost_hypgeom.boost_hypergeometric_pfq_batch_fixed_point(pfq_a, pfq_b, z_mid),
        ),
    ]

    lines = [
        "Last updated: 2026-03-08T00:00:00Z",
        "",
        "# Hypgeom Point Kernel Benchmark",
        "",
        "CPU benchmark for the heavy hypergeometric point kernels after the point/basic kernel split cleanup.",
        "",
        "| family | point_api_ms | point_kernel_ms |",
        "|---|---:|---:|",
    ]
    for family, api_point_fn, kernel_point_fn in rows:
        point_api_ms = _timeit(api_point_fn)
        point_kernel_ms = _timeit(kernel_point_fn)
        lines.append(f"| {family} | {point_api_ms:.4f} | {point_kernel_ms:.4f} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
