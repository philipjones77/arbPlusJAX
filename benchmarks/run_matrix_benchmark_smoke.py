from __future__ import annotations

import time
from pathlib import Path

import jax.numpy as jnp

from arbplusjax import acb_mat
from arbplusjax import arb_mat
from arbplusjax import mat_wrappers


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "reports" / "matrix_mode_benchmark_smoke.md"


def _bench(fn, *args, repeat: int = 20):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat


def main() -> None:
    a_real = jnp.array(
        [
            [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0], [4.0, 4.0]],
        ],
        dtype=jnp.float64,
    )
    a_cplx = jnp.array(
        [
            [[2.0, 2.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 0.5, 0.5]],
        ],
        dtype=jnp.float64,
    )

    rows = [
        ("arb_mat_det", "point", _bench(arb_mat.arb_mat_det_jit, a_real)),
        ("arb_mat_det", "basic", _bench(arb_mat.arb_mat_det_basic, a_real)),
        ("arb_mat_det", "rigorous", _bench(lambda x: mat_wrappers.arb_mat_det_mode(x, impl="rigorous", prec_bits=53), a_real)),
        ("arb_mat_trace", "point", _bench(arb_mat.arb_mat_trace_jit, a_real)),
        ("arb_mat_trace", "basic", _bench(arb_mat.arb_mat_trace_basic, a_real)),
        ("arb_mat_trace", "rigorous", _bench(lambda x: mat_wrappers.arb_mat_trace_mode(x, impl="rigorous", prec_bits=53), a_real)),
        ("acb_mat_det", "point", _bench(acb_mat.acb_mat_det_jit, a_cplx)),
        ("acb_mat_det", "basic", _bench(acb_mat.acb_mat_det_basic, a_cplx)),
        ("acb_mat_det", "rigorous", _bench(lambda x: mat_wrappers.acb_mat_det_mode(x, impl="rigorous", prec_bits=53), a_cplx)),
        ("acb_mat_trace", "point", _bench(acb_mat.acb_mat_trace_jit, a_cplx)),
        ("acb_mat_trace", "basic", _bench(acb_mat.acb_mat_trace_basic, a_cplx)),
        ("acb_mat_trace", "rigorous", _bench(lambda x: mat_wrappers.acb_mat_trace_mode(x, impl="rigorous", prec_bits=53), a_cplx)),
    ]

    lines = [
        "Last updated: 2026-03-07T00:00:00Z",
        "",
        "# Matrix Mode Benchmark Smoke",
        "",
        "Quick warmed benchmark for canonical `arb_mat` / `acb_mat` `n x n` determinant and trace paths.",
        "",
        "| function | mode | mean_time_ms |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {mode} | {ms:.4f} |" for name, mode, ms in rows)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
