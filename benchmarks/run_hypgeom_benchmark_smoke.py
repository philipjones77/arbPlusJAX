from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import api, double_interval as di, hypgeom


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "reports" / "hypgeom_mode_benchmark_smoke.md"


def _iv_batch(lo0: float, hi0: float, n: int) -> jax.Array:
    lo = jnp.linspace(jnp.float64(lo0), jnp.float64(hi0), n)
    return di.interval(lo, lo + jnp.float64(0.05))


def _timeit(fn, *args, repeats: int = 3, **kwargs):
    fn(*args, **kwargs)
    start = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    elapsed = (time.perf_counter() - start) / repeats * 1e3
    return elapsed


def _manual_padded(batch_fn, trim_n: int, *args, pad_to: int, static_prefix: tuple = ()):
    padded = []
    for arg in args:
        pad_count = pad_to - arg.shape[0]
        pad = jnp.repeat(arg[-1:], pad_count, axis=0)
        padded.append(jnp.concatenate([arg, pad], axis=0))
    out = batch_fn(*static_prefix, *padded)
    return out[:trim_n]


def main() -> None:
    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Hypgeom Mode Benchmark Smoke",
        "",
        "Small CPU smoke benchmark for selected canonical and alternative hypgeom families.",
        "",
        "Comparisons:",
        "- `unpadded_basic_ms`: current batch path",
        "- `padded_basic_ms`: fixed-shape padded batch path",
        "",
        "| family | path | unpadded_basic_ms | padded_basic_ms |",
        "|---|---|---:|---:|",
    ]

    n = 8
    pad_to = 16
    gamma_x = _iv_batch(0.9, 1.3, n)
    erf_x = _iv_batch(0.1, 0.5, n)
    ei_x = _iv_batch(0.2, 0.6, n)
    g_s = _iv_batch(1.1, 1.4, n)
    g_z = _iv_batch(0.2, 0.6, n)
    a = _iv_batch(1.1, 1.4, n)
    b = _iv_batch(2.1, 2.4, n)
    b2 = _iv_batch(1.2, 1.5, n)
    c = _iv_batch(2.1, 2.4, n)
    z01 = _iv_batch(0.1, 0.4, n)
    z02 = _iv_batch(0.1, 0.2, n)
    z03 = _iv_batch(0.1, 0.3, n)
    uz = _iv_batch(0.6, 1.2, n)
    leg_m = _iv_batch(0.0, 0.0, n)
    pfq_a = jnp.stack((_iv_batch(0.6, 0.8, n)[..., 0], _iv_batch(0.9, 1.1, n)[..., 0]), axis=1)
    pfq_b = jnp.stack((_iv_batch(1.4, 1.6, n)[..., 0],), axis=1)
    rows = [
        ("arb_hypgeom_gamma", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma", gamma_x, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma", gamma_x, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_erf", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_erf", erf_x, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_erf", erf_x, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_ei", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_ei", ei_x, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_ei", ei_x, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_gamma_lower", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma_lower", g_s, g_z, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma_lower", g_s, g_z, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_gamma_upper", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma_upper", g_s, g_z, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gamma_upper", g_s, g_z, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_0f1", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_0f1", a, z01, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_0f1", a, z01, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_1f1", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_1f1", a, b, z01, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_1f1", a, b, z01, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_2f1", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_2f1", a, b2, c, z02, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_2f1", a, b2, c, z02, mode="basic", pad_to=pad_to)),
        ("arb_hypgeom_u", lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_u", a, b, uz, mode="basic"), lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_u", a, b, uz, mode="basic", pad_to=pad_to)),
        (
            "arb_hypgeom_pfq",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_pfq", pfq_a, pfq_b, z03, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_pfq", pfq_a, pfq_b, z03, mode="basic", pad_to=pad_to),
        ),
        ("boost_hypergeometric_1f1", lambda: api.eval_interval_batch("boost_hypgeom.boost_hypergeometric_1f1", a, b, z01, mode="basic"), lambda: api.eval_interval_batch("boost_hypgeom.boost_hypergeometric_1f1", a, b, z01, mode="basic", pad_to=pad_to)),
        ("cusf_hyp1f1", lambda: api.eval_interval_batch("cusf_compat.cusf_hyp1f1", a, b, z01, mode="basic"), lambda: api.eval_interval_batch("cusf_compat.cusf_hyp1f1", a, b, z01, mode="basic", pad_to=pad_to)),
    ]
    for family, unpadded_fn, padded_fn in rows:
        unpadded_ms = _timeit(unpadded_fn)
        padded_ms = _timeit(padded_fn)
        lines.append(f"| {family} | api.eval_interval_batch | {unpadded_ms:.4f} | {padded_ms:.4f} |")

    z = _iv_batch(0.1, 0.5, n)
    z_small = z
    za = _iv_batch(0.1, 0.2, n)
    zb = _iv_batch(0.2, 0.3, n)
    zlam = _iv_batch(0.6, 0.8, n)
    rows2 = [
        (
            "arb_hypgeom_legendre_p",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_legendre_p", 2, leg_m, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_legendre_p", 2, leg_m, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_jacobi_p",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_jacobi_p", 2, za, zb, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_jacobi_p", 2, za, zb, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_gegenbauer_c",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gegenbauer_c", 2, zlam, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_gegenbauer_c", 2, zlam, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_chebyshev_t",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_chebyshev_t", 2, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_chebyshev_t", 2, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_chebyshev_u",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_chebyshev_u", 2, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_chebyshev_u", 2, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_laguerre_l",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_laguerre_l", 2, za, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_laguerre_l", 2, za, z_small, mode="basic", pad_to=pad_to),
        ),
        (
            "arb_hypgeom_hermite_h",
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_hermite_h", 2, z_small, mode="basic"),
            lambda: api.eval_interval_batch("hypgeom.arb_hypgeom_hermite_h", 2, z_small, mode="basic", pad_to=pad_to),
        ),
    ]
    for family, unpadded_fn, padded_fn in rows2:
        unpadded_ms = _timeit(unpadded_fn)
        padded_ms = _timeit(padded_fn)
        lines.append(f"| {family} | api.eval_interval_batch | {unpadded_ms:.4f} | {padded_ms:.4f} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
