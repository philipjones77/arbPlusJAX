from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

_FAMILY_COMPILE_MARKERS = {
    "boost_hypergeometric_0f1": ("boost_hypergeometric_0f1", "arb_hypgeom_0f1", "acb_hypgeom_0f1"),
    "boost_hypergeometric_1f1": ("boost_hypergeometric_1f1", "arb_hypgeom_1f1", "acb_hypgeom_1f1"),
    "boost_hyp2f1_series": ("boost_hyp2f1_series", "arb_hypgeom_2f1", "acb_hypgeom_2f1"),
    "boost_hyp2f1_cf": ("boost_hyp2f1_cf", "arb_hypgeom_2f1", "acb_hypgeom_2f1"),
    "boost_hyp2f1_pade": ("boost_hyp2f1_pade", "arb_hypgeom_2f1", "acb_hypgeom_2f1"),
    "boost_hyp2f1_rational": ("boost_hyp2f1_rational", "arb_hypgeom_2f1", "acb_hypgeom_2f1"),
    "boost_hypergeometric_pfq": ("boost_hypergeometric_pfq", "arb_hypgeom_pfq", "acb_hypgeom_pfq"),
}


def _worker(functions: list[str], sizes: list[int], pad_to: int | None, mode: str, dtype: str) -> None:
    import jax.numpy as jnp
    import numpy as np
    from arbplusjax import api, double_interval as di

    dt = jnp.float32 if dtype == "float32" else jnp.float64

    def mk_interval(n: int, lo: float, hi: float):
        a = jnp.asarray(np.linspace(lo, hi, n), dtype=dt)
        return di.interval(a, a + jnp.asarray(0.05, dtype=dt))

    def pad_last(arg, target: int):
        arr = jnp.asarray(arg)
        if arr.shape[0] == target:
            return arr
        pad_count = target - int(arr.shape[0])
        pad_block = jnp.repeat(arr[-1:, ...], pad_count, axis=0)
        return jnp.concatenate((arr, pad_block), axis=0)

    for n in sizes:
        a = mk_interval(n, 1.1, 1.3)
        b = mk_interval(n, 2.1, 2.3)
        c = mk_interval(n, 2.8, 3.0)
        z = mk_interval(n, 0.1, 0.3)
        u_z = mk_interval(n, 0.6, 1.0)
        m = mk_interval(n, 0.0, 0.0)
        lam = mk_interval(n, 0.6, 0.8)
        pfq_a = jnp.stack((mk_interval(n, 0.6, 0.8)[..., 0], mk_interval(n, 0.9, 1.1)[..., 0]), axis=1)
        pfq_b = jnp.stack((mk_interval(n, 1.4, 1.6)[..., 0],), axis=1)
        if pad_to is not None:
            a = pad_last(a, pad_to)
            b = pad_last(b, pad_to)
            c = pad_last(c, pad_to)
            z = pad_last(z, pad_to)
            u_z = pad_last(u_z, pad_to)
            m = pad_last(m, pad_to)
            lam = pad_last(lam, pad_to)
            pfq_a = pad_last(pfq_a, pad_to)
            pfq_b = pad_last(pfq_b, pad_to)
        for name in functions:
            if name == "arb_hypgeom_0f1":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_gamma_lower":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_gamma_upper":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_1f1":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, b, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_2f1":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, b, c, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_u":
                out = api.eval_interval_batch(f"hypgeom.{name}", a, b, u_z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_legendre_p":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, m, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_jacobi_p":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, a, b, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_gegenbauer_c":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, lam, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_chebyshev_t":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_chebyshev_u":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_laguerre_l":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, a, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_hermite_h":
                out = api.eval_interval_batch(f"hypgeom.{name}", 2, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "arb_hypgeom_pfq":
                out = api.eval_interval_batch(f"hypgeom.{name}", pfq_a, pfq_b, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "boost_hypergeometric_0f1":
                out = api.eval_interval_batch(name, a, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "boost_hypergeometric_1f1":
                out = api.eval_interval_batch(name, a, b, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name in ("boost_hyp2f1_series", "boost_hyp2f1_cf", "boost_hyp2f1_pade", "boost_hyp2f1_rational"):
                out = api.eval_interval_batch(name, a, b, c, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            elif name == "boost_hypergeometric_pfq":
                out = api.eval_interval_batch(name, pfq_a, pfq_b, z, mode=mode, pad_to=pad_to, dtype=dtype, prec_bits=53)
            else:
                raise KeyError(name)
            _ = jnp.asarray(out).block_until_ready()


def _run_once(functions: list[str], sizes: list[int], pad_to: int | None, mode: str, dtype: str) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["JAX_LOG_COMPILES"] = "1"
    env.setdefault("JAX_PLATFORMS", "cpu")
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--functions",
        ",".join(functions),
        "--sizes",
        ",".join(str(s) for s in sizes),
        "--mode",
        mode,
        "--dtype",
        dtype,
    ]
    if pad_to is not None:
        args.extend(["--pad-to", str(pad_to)])
    proc = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    compile_lines = [line for line in proc.stderr.splitlines() if "Compiling jit(" in line]
    family_markers = tuple(marker for name in functions for marker in _FAMILY_COMPILE_MARKERS.get(name, (name,)))
    family_compile_lines = [line for line in compile_lines if any(marker in line for marker in family_markers)]
    return {
        "functions": functions,
        "sizes": sizes,
        "pad_to": pad_to,
        "mode": mode,
        "dtype": dtype,
        "compile_events": len(compile_lines),
        "family_compile_events": len(family_compile_lines),
        "compile_lines": compile_lines,
        "family_compile_lines": family_compile_lines,
    }


def _render_md(unpadded: dict, padded: dict) -> str:
    return "\n".join(
        [
            "Last updated: 2026-03-07T00:00:00Z",
            "",
            "# Hypgeom Compile Probe",
            "",
            "Generated by `tools/hypgeom_compile_probe.py`.",
            "",
            f"- functions: `{','.join(unpadded['functions'])}`",
            f"- sizes: `{','.join(str(s) for s in unpadded['sizes'])}`",
            f"- mode: `{unpadded['mode']}`",
            f"- dtype: `{unpadded['dtype']}`",
            f"- unpadded compile events: `{unpadded['compile_events']}`",
            f"- caller-fixed compile events: `{padded['compile_events']}`",
            f"- unpadded family compile events: `{unpadded['family_compile_events']}`",
            f"- caller-fixed family compile events: `{padded['family_compile_events']}`",
            f"- fixed target: `{padded['pad_to']}`",
            "",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--functions",
        default="arb_hypgeom_0f1,arb_hypgeom_gamma_lower,arb_hypgeom_gamma_upper,arb_hypgeom_1f1,arb_hypgeom_2f1,arb_hypgeom_u,arb_hypgeom_legendre_p,arb_hypgeom_jacobi_p,arb_hypgeom_gegenbauer_c,arb_hypgeom_chebyshev_t,arb_hypgeom_chebyshev_u,arb_hypgeom_laguerre_l,arb_hypgeom_hermite_h,arb_hypgeom_pfq",
    )
    parser.add_argument("--sizes", default="20,40")
    parser.add_argument("--pad-to", type=int)
    parser.add_argument("--mode", choices=("basic", "adaptive", "rigorous"), default="basic")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--out-json", default="results/benchmarks/hypgeom_compile_probe/hypgeom_compile_probe.json")
    parser.add_argument("--out-md", default="docs/status/reports/hypgeom_compile_probe.md")
    args = parser.parse_args()

    functions = [x for x in args.functions.split(",") if x]
    sizes = [int(x) for x in args.sizes.split(",") if x]
    if args.worker:
        _worker(functions, sizes, args.pad_to, args.mode, args.dtype)
        return

    unpadded = _run_once(functions, sizes, None, args.mode, args.dtype)
    padded = _run_once(functions, sizes, max(sizes) if args.pad_to is None else args.pad_to, args.mode, args.dtype)
    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"unpadded": unpadded, "padded": padded}, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(unpadded, padded), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
