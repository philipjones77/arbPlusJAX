from __future__ import annotations

import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from arbplusjax import api


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "reports" / "incomplete_bessel_benchmark.md"


def _timeit(fn, *args, repeats: int = 5, **kwargs) -> tuple[float, object]:
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    elapsed_ms = (time.perf_counter() - start) / repeats * 1e3
    return elapsed_ms, out


def _backend_summary() -> str:
    devices = jax.devices()
    kinds = sorted({d.platform for d in devices})
    return ", ".join(kinds) if kinds else "unknown"


def main() -> None:
    nu = jnp.float64(0.5)
    lower_small = jnp.float64(0.05)
    lower_mid = jnp.float64(0.4)
    lower_large = jnp.float64(1.2)
    z_mid = jnp.float64(1.4)
    z_large = jnp.float64(20.0)
    z_fragile = jnp.float64(0.5)
    upper = jnp.float64(1.2)

    rows: list[tuple[str, str, callable]] = [
        (
            "incomplete_bessel_k",
            "quadrature_benign",
            lambda: api.incomplete_bessel_k(nu, z_mid, lower_mid, mode="point", method="quadrature"),
        ),
        (
            "incomplete_bessel_k",
            "recurrence_large_lower",
            lambda: api.incomplete_bessel_k(nu, z_large, lower_large, mode="point", method="recurrence"),
        ),
        (
            "incomplete_bessel_k",
            "asymptotic_large_decay",
            lambda: api.incomplete_bessel_k(nu, z_large, lower_mid, mode="point", method="asymptotic"),
        ),
        (
            "incomplete_bessel_k",
            "mpfallback_fragile",
            lambda: api.incomplete_bessel_k(jnp.float64(13.0), z_fragile, lower_small, mode="point", method="mpfallback"),
        ),
        (
            "incomplete_bessel_i",
            "angular_point",
            lambda: api.incomplete_bessel_i(jnp.float64(1.0), jnp.float64(0.8), upper, mode="point"),
        ),
        (
            "incomplete_bessel_i",
            "angular_basic",
            lambda: api.incomplete_bessel_i(jnp.float64(1.0), jnp.float64(0.8), upper, mode="basic"),
        ),
        (
            "incomplete_bessel_i",
            "angular_batch_point",
            lambda: api.incomplete_bessel_i_batch(
                jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
                jnp.asarray([1.0, 0.8, 0.6], dtype=jnp.float64),
                jnp.asarray([jnp.pi, 1.2, 0.9], dtype=jnp.float64),
                mode="point",
            ),
        ),
    ]

    lines = [
        "Last updated: 2026-03-13T00:00:00Z",
        "",
        "# Incomplete Bessel Benchmark",
        "",
        "Benchmark summary for the current incomplete-Bessel package and tail-engine-backed method surface.",
        "",
        "## Environment",
        "",
        f"- OS: {platform.platform()}",
        f"- Backend: {jax.default_backend()}",
        f"- Devices: {_backend_summary()}",
        f"- Default dtype: float64 / {jnp.complex128}",
        "",
        "| family | regime | time_ms | sample_output |",
        "|---|---|---:|---:|",
    ]

    for family, regime, fn in rows:
        time_ms, out = _timeit(fn)
        sample = jnp.ravel(jnp.asarray(out))[0]
        lines.append(f"| {family} | {regime} | {time_ms:.4f} | {float(sample):.6g} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
