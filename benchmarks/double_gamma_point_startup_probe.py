from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env


def _run_import_probe(module_expr: str) -> dict[str, float]:
    code = (
        "import time\n"
        "t0=time.perf_counter()\n"
        f"{module_expr}\n"
        "print(time.perf_counter()-t0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_base_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    seconds = float(completed.stdout.strip().splitlines()[-1])
    return {"seconds": round(seconds, 6)}


def _run_worker(family: str, pad_to: int) -> dict[str, object]:
    env = _base_env()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", "--family", family, "--pad-to", str(pad_to)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _worker(family: str, pad_to: int) -> None:
    import jax
    import jax.numpy as jnp

    from arbplusjax import api
    from arbplusjax import stable_kernels

    if family not in {"bdg_barnesgamma2", "ifj_barnesdoublegamma", "provider_barnesdoublegamma"}:
        raise ValueError(f"unsupported family {family!r}")

    w = jnp.array([1.2 + 0.1j, 1.4 - 0.2j], dtype=jnp.complex64)
    beta = jnp.array([0.9 + 0.0j, 1.1 + 0.1j], dtype=jnp.complex64)
    z = jnp.array([1.2 + 0.1j, 1.4 - 0.2j], dtype=jnp.complex128)
    tau = jnp.array([0.8, 1.0], dtype=jnp.float64)

    t0 = time.perf_counter()
    _ = jax.devices()
    t1 = time.perf_counter()

    payload: dict[str, object] = {
        "family": family,
        "pad_to": pad_to,
        "backend_init_s": round(t1 - t0, 6),
    }

    try:
        if family == "bdg_barnesgamma2":
            bound = api.bind_point_batch_jit(family, dtype="float32", pad_to=pad_to)
            args = (w, beta)
        elif family == "ifj_barnesdoublegamma":
            bound = api.bind_point_batch_jit(
                family,
                pad_to=pad_to,
                dps=60,
                max_m_cap=128,
            )
            args = (z, tau)
        else:
            bound = lambda zz, tt: stable_kernels.provider_barnesdoublegamma_batch(
                zz,
                tt,
                pad_to=pad_to,
                dps=60,
            )
            args = (z, tau)
        t2 = time.perf_counter()
        out = bound(*args)
        jax.block_until_ready(out)
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        out = bound(*args)
        jax.block_until_ready(out)
        t5 = time.perf_counter()

        payload.update(
            {
                "compile_plus_first_point_batch_s": round(t3 - t2, 6),
                "steady_point_batch_s": round(t5 - t4, 6),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "compile_plus_first_point_batch_s": None,
                "steady_point_batch_s": None,
                "compile_error_type": type(exc).__name__,
                "compile_error": str(exc).splitlines()[0],
            }
        )

    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--family",
        choices=("bdg_barnesgamma2", "ifj_barnesdoublegamma", "provider_barnesdoublegamma"),
        default="ifj_barnesdoublegamma",
    )
    parser.add_argument("--pad-to", type=int, default=4)
    parser.add_argument(
        "--out-json",
        default="benchmarks/results/double_gamma_point_startup_probe/double_gamma_point_startup_probe.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmarks/results/double_gamma_point_startup_probe/double_gamma_point_startup_probe.md",
    )
    args = parser.parse_args()

    if args.worker:
        _worker(args.family, args.pad_to)
        return

    payload = {"import_arbplusjax_api": _run_import_probe("from arbplusjax import api")}
    families = ("bdg_barnesgamma2", "ifj_barnesdoublegamma", "provider_barnesdoublegamma") if args.family == "ifj_barnesdoublegamma" else (args.family,)
    for family in families:
        payload[f"{family}_point_path"] = _run_worker(family, args.pad_to)

    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    first_key = next(key for key in payload if key.endswith("_point_path"))
    first = payload[first_key]
    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Double Gamma Point Startup Probe",
        "",
        "Generated by `benchmarks/double_gamma_point_startup_probe.py`.",
        "",
        f"- api import s: `{payload['import_arbplusjax_api']['seconds']}`",
        f"- family: `{first['family']}`",
        f"- pad_to: `{first['pad_to']}`",
        f"- backend init s: `{first['backend_init_s']}`",
        f"- compile plus first point batch s: `{first['compile_plus_first_point_batch_s']}`",
        f"- steady point batch s: `{first['steady_point_batch_s']}`",
    ]
    if first.get("compile_error_type"):
        lines.append(f"- compile error type: `{first['compile_error_type']}`")
        lines.append(f"- compile error: `{first['compile_error']}`")
    for key, row in payload.items():
        if key == "import_arbplusjax_api":
            continue
        lines.extend(
            [
                "",
                f"## {row['family']}",
                f"- backend init s: `{row['backend_init_s']}`",
                f"- compile plus first point batch s: `{row['compile_plus_first_point_batch_s']}`",
                f"- steady point batch s: `{row['steady_point_batch_s']}`",
            ]
        )
        if row.get("compile_error_type"):
            lines.append(f"- compile error type: `{row['compile_error_type']}`")
            lines.append(f"- compile error: `{row['compile_error']}`")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
