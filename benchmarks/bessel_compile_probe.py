from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker(functions: list[str], sizes: list[int], pad_to: int | None, dtype: str) -> None:
    import jax.numpy as jnp
    from arbplusjax import hypgeom

    dt = jnp.float32 if dtype == "float32" else jnp.float64

    def _pad_rows_to(x, target: int):
        pad = max(target - int(x.shape[0]), 0)
        if pad == 0:
            return x
        tail = jnp.repeat(x[-1:, :], pad, axis=0)
        return jnp.concatenate([x, tail], axis=0)

    def mk_interval(n: int, lo: float, hi: float):
        a = jnp.linspace(jnp.asarray(lo, dtype=dt), jnp.asarray(hi, dtype=dt), n)
        b = a + jnp.asarray(0.05, dtype=dt)
        return jnp.stack([a, b], axis=1)

    def mk_box(n: int, re_lo: float, re_hi: float, im_lo: float, im_hi: float):
        re0 = jnp.linspace(jnp.asarray(re_lo, dtype=dt), jnp.asarray(re_hi, dtype=dt), n)
        re1 = re0 + jnp.asarray(0.05, dtype=dt)
        im0 = jnp.linspace(jnp.asarray(im_lo, dtype=dt), jnp.asarray(im_hi, dtype=dt), n)
        im1 = im0 + jnp.asarray(0.05, dtype=dt)
        return jnp.stack([re0, re1, im0, im1], axis=1)

    real_map = {
        "besselj": hypgeom.arb_hypgeom_bessel_j_batch_prec_jit,
        "bessely": hypgeom.arb_hypgeom_bessel_y_batch_prec_jit,
        "besseli": hypgeom.arb_hypgeom_bessel_i_batch_prec_jit,
        "besselk": hypgeom.arb_hypgeom_bessel_k_batch_prec_jit,
        "besseli_scaled": hypgeom.arb_hypgeom_bessel_i_scaled_batch_prec_jit,
        "besselk_scaled": hypgeom.arb_hypgeom_bessel_k_scaled_batch_prec_jit,
    }
    real_pad_map = {
        "besselj": hypgeom.arb_hypgeom_bessel_j_batch_fixed_prec,
        "bessely": hypgeom.arb_hypgeom_bessel_y_batch_fixed_prec,
        "besseli": hypgeom.arb_hypgeom_bessel_i_batch_fixed_prec,
        "besselk": hypgeom.arb_hypgeom_bessel_k_batch_fixed_prec,
        "besseli_scaled": hypgeom.arb_hypgeom_bessel_i_scaled_batch_fixed_prec,
        "besselk_scaled": hypgeom.arb_hypgeom_bessel_k_scaled_batch_fixed_prec,
    }
    complex_map = {
        "acb_besselj": hypgeom.acb_hypgeom_bessel_j_batch_prec_jit,
        "acb_bessely": hypgeom.acb_hypgeom_bessel_y_batch_prec_jit,
        "acb_besseli": hypgeom.acb_hypgeom_bessel_i_batch_prec_jit,
        "acb_besselk": hypgeom.acb_hypgeom_bessel_k_batch_prec_jit,
        "acb_besseli_scaled": hypgeom.acb_hypgeom_bessel_i_scaled_batch_prec_jit,
        "acb_besselk_scaled": hypgeom.acb_hypgeom_bessel_k_scaled_batch_prec_jit,
    }
    complex_pad_map = {
        "acb_besselj": hypgeom.acb_hypgeom_bessel_j_batch_fixed_prec,
        "acb_bessely": hypgeom.acb_hypgeom_bessel_y_batch_fixed_prec,
        "acb_besseli": hypgeom.acb_hypgeom_bessel_i_batch_fixed_prec,
        "acb_besselk": hypgeom.acb_hypgeom_bessel_k_batch_fixed_prec,
        "acb_besseli_scaled": hypgeom.acb_hypgeom_bessel_i_scaled_batch_fixed_prec,
        "acb_besselk_scaled": hypgeom.acb_hypgeom_bessel_k_scaled_batch_fixed_prec,
    }

    for n in sizes:
        nu_r = mk_interval(n, 0.2, 0.6)
        z_r = mk_interval(n, 1.5, 3.5)
        nu_c = mk_box(n, 0.2, 0.6, -0.1, 0.1)
        z_c = mk_box(n, 1.5, 3.5, -0.2, 0.2)
        for name in functions:
            if name in real_map:
                if pad_to is None:
                    out = real_map[name](nu_r, z_r, prec_bits=80, mode="sample")
                else:
                    out = real_pad_map[name](_pad_rows_to(nu_r, pad_to), _pad_rows_to(z_r, pad_to), prec_bits=80, mode="sample")
            else:
                if pad_to is None:
                    out = complex_map[name](nu_c, z_c, prec_bits=80)
                else:
                    out = complex_pad_map[name](_pad_rows_to(nu_c, pad_to), _pad_rows_to(z_c, pad_to), prec_bits=80)
            _ = jnp.asarray(out).block_until_ready()


def _run_once(functions: list[str], sizes: list[int], pad_to: int | None, dtype: str) -> dict:
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
        "--dtype",
        dtype,
    ]
    if pad_to is not None:
        args.extend(["--pad-to", str(pad_to)])
    proc = subprocess.run(args, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True)
    compile_lines = [line for line in proc.stderr.splitlines() if "Compiling jit(" in line]
    core_markers = (
        "arb_hypgeom_bessel_",
        "acb_hypgeom_bessel_",
        "_arb_hypgeom_bessel_",
        "_acb_hypgeom_bessel_",
    )
    core_compile_lines = [line for line in compile_lines if any(marker in line for marker in core_markers)]
    return {
        "functions": functions,
        "sizes": sizes,
        "pad_to": pad_to,
        "dtype": dtype,
        "compile_events": len(compile_lines),
        "core_compile_events": len(core_compile_lines),
        "compile_lines": compile_lines,
        "core_compile_lines": core_compile_lines,
    }


def _render_md(unpadded: dict, padded: dict) -> str:
    return "\n".join(
        [
            "Last updated: 2026-03-07T00:00:00Z",
            "",
            "# Bessel Compile Probe",
            "",
            "Generated by `benchmarks/bessel_compile_probe.py`.",
            "",
            f"- functions: `{','.join(unpadded['functions'])}`",
            f"- sizes: `{','.join(str(s) for s in unpadded['sizes'])}`",
            f"- dtype: `{unpadded['dtype']}`",
            f"- unpadded compile events: `{unpadded['compile_events']}`",
            f"- caller-padded fixed-shape compile events: `{padded['compile_events']}`",
            f"- unpadded Bessel-core compile events: `{unpadded['core_compile_events']}`",
            f"- caller-padded fixed-shape Bessel-core compile events: `{padded['core_compile_events']}`",
            f"- padded target: `{padded['pad_to']}`",
            "",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--functions", default="besselj,bessely,besseli,besselk,besseli_scaled,besselk_scaled")
    parser.add_argument("--sizes", default="40,80")
    parser.add_argument("--pad-to", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--outdir", default="benchmarks/results/bessel_compile_probe")
    args = parser.parse_args()

    functions = [x for x in args.functions.split(",") if x]
    sizes = [int(x) for x in args.sizes.split(",") if x]
    if args.worker:
        _worker(functions, sizes, args.pad_to, args.dtype)
        return

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    unpadded = _run_once(functions, sizes, None, args.dtype)
    padded = _run_once(functions, sizes, max(sizes) if args.pad_to is None else args.pad_to, args.dtype)
    (outdir / "bessel_compile_probe.json").write_text(
        json.dumps({"unpadded": unpadded, "padded": padded}, indent=2) + "\n",
        encoding="utf-8",
    )
    (outdir / "bessel_compile_probe.md").write_text(_render_md(unpadded, padded), encoding="utf-8")
    print(f"Wrote: {outdir / 'bessel_compile_probe.json'}")
    print(f"Wrote: {outdir / 'bessel_compile_probe.md'}")


if __name__ == "__main__":
    main()
