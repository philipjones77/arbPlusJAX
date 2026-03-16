from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

from arbplusjax import double_interval as di
from arbplusjax import jrb_mat


REPO_ROOT = Path(__file__).resolve().parents[1]


def _point_interval(x: jax.Array) -> jax.Array:
    xx = jnp.asarray(x)
    return di.interval(xx, xx)


def _rademacher_probes(key: jax.Array, num_probes: int, n: int, dtype=jnp.float64) -> jax.Array:
    mids = random.rademacher(key, (num_probes, n), dtype=dtype)
    return jax.vmap(_point_interval)(mids)


def _dense_operator(a: jax.Array):
    aa = jnp.asarray(a)
    return jrb_mat.jrb_mat_dense_operator(di.interval(aa, aa))


def _make_spd(key: jax.Array, n: int, cond: float = 1e2, dtype=jnp.float64) -> jax.Array:
    q, _ = jnp.linalg.qr(random.normal(key, (n, n), dtype=dtype))
    eigs = jnp.exp(jnp.linspace(jnp.log(jnp.asarray(1.0 / cond, dtype=dtype)), 0.0, n, dtype=dtype))
    return (q.T * eigs) @ q


def main() -> int:
    outdir = REPO_ROOT / "outputs" / "metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    exactness = []
    for n in (4, 8):
        diag = jnp.exp(jnp.linspace(jnp.log(0.01), 0.0, n, dtype=jnp.float64))
        op = _dense_operator(jnp.diag(diag))
        probes = _rademacher_probes(random.PRNGKey(n), 16, n)
        est = jrb_mat.jrb_mat_logdet_slq_point(op, probes, n)
        truth = jnp.sum(jnp.log(diag))
        exactness.append(
            {
                "n": n,
                "estimate": float(est),
                "truth": float(truth),
                "abs_error": float(jnp.abs(est - truth)),
            }
        )

    diag = jnp.asarray([0.2, 0.4, 0.8, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=jnp.float64)
    base = jrb_mat.jrb_mat_logdet_slq_point(_dense_operator(jnp.diag(diag)), _rademacher_probes(random.PRNGKey(0), 16, 8), 8)
    tail = []
    for factor in (0.5, 2.0):
        scaled = diag.at[jnp.asarray([6, 7], dtype=jnp.int32)].multiply(factor)
        est = jrb_mat.jrb_mat_logdet_slq_point(_dense_operator(jnp.diag(scaled)), _rademacher_probes(random.PRNGKey(0), 16, 8), 8)
        expected_delta = 2.0 * jnp.log(jnp.asarray(factor, dtype=jnp.float64))
        tail.append(
            {
                "factor": float(factor),
                "delta_estimate": float(est - base),
                "delta_truth": float(expected_delta),
                "abs_error": float(jnp.abs((est - base) - expected_delta)),
            }
        )

    a = _make_spd(random.PRNGKey(11), 8, cond=1e2)
    op = _dense_operator(a)
    budget = []
    for probes_n in (4, 8, 16, 32):
        vals = []
        for seed in range(8):
            probes = _rademacher_probes(random.PRNGKey(seed), probes_n, 8)
            vals.append(jrb_mat.jrb_mat_logdet_slq_point(op, probes, 8))
        vals = jnp.asarray(vals, dtype=jnp.float64)
        budget.append(
            {
                "num_probes": probes_n,
                "mean": float(jnp.mean(vals)),
                "std": float(jnp.std(vals)),
                "median": float(jnp.median(vals)),
            }
        )

    diag64 = jnp.asarray([0.3, 0.75, 1.0, 1.8, 3.0, 5.0], dtype=jnp.float64)
    key = random.PRNGKey(123)
    est64 = jrb_mat.jrb_mat_logdet_slq_point(
        _dense_operator(jnp.diag(diag64)),
        _rademacher_probes(key, 16, diag64.shape[0], dtype=jnp.float64),
        diag64.shape[0],
    )
    est64_repeat = jrb_mat.jrb_mat_logdet_slq_point(
        _dense_operator(jnp.diag(diag64)),
        _rademacher_probes(key, 16, diag64.shape[0], dtype=jnp.float64),
        diag64.shape[0],
    )
    est32 = jrb_mat.jrb_mat_logdet_slq_point(
        _dense_operator(jnp.diag(diag64.astype(jnp.float32))),
        _rademacher_probes(key, 16, diag64.shape[0], dtype=jnp.float32),
        diag64.shape[0],
    )

    payload = {
        "exactness": exactness,
        "tail_sensitivity": tail,
        "budget": budget,
        "reproducibility": {
            "estimate64": float(est64),
            "estimate64_repeat": float(est64_repeat),
            "estimate32": float(est32),
            "repeat_abs_diff": float(jnp.abs(est64 - est64_repeat)),
            "dtype_abs_diff": float(jnp.abs(est64 - est32.astype(jnp.float64))),
        },
    }

    out = outdir / "slq_logdet_contracts.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
