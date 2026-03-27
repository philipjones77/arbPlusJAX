from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from _source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_RUNNERS = (
    "incomplete_gamma_upper",
    "hypgeom_1f1",
    "hypgeom_u",
    "barnesdoublegamma_ifj",
    "bessel_j",
)


def _time_call(fn, *args, iters: int = 1):
    import jax

    started = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return out, (time.perf_counter() - started) / float(iters)


def _bench_gamma(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import api

    s = jnp.float64(2.5)
    z = jnp.float64(1.75)
    import jax

    arg_grad = jax.grad(lambda zv: api.incomplete_gamma_upper(s, zv, mode="point", method="quadrature"))
    param_grad = jax.grad(lambda sv: api.incomplete_gamma_upper(sv, z, mode="point", method="quadrature"))
    _, arg_s = _time_call(arg_grad, z, iters=iters)
    _, param_s = _time_call(param_grad, s, iters=iters)
    return {"argument_grad_s": round(arg_s, 6), "parameter_grad_s": round(param_s, 6)}


def _bench_hypgeom_1f1(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import api

    a = jnp.float64(1.25)
    b = jnp.float64(2.25)
    z = jnp.float64(0.3)
    import jax

    arg_grad = jax.grad(lambda zv: api.eval_point("hypgeom.arb_hypgeom_1f1", a, b, zv))
    param_grad = jax.grad(lambda av: api.eval_point("hypgeom.arb_hypgeom_1f1", av, b, z))
    _, arg_s = _time_call(arg_grad, z, iters=iters)
    _, param_s = _time_call(param_grad, a, iters=iters)
    return {"argument_grad_s": round(arg_s, 6), "parameter_grad_s": round(param_s, 6)}


def _bench_hypgeom_u(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import api

    a = jnp.float64(1.0)
    b = jnp.float64(1.5)
    z = jnp.float64(0.2)
    import jax

    arg_grad = jax.grad(lambda zv: api.eval_point("hypgeom.arb_hypgeom_u", a, b, zv))
    param_grad = jax.grad(lambda av: api.eval_point("hypgeom.arb_hypgeom_u", av, b, z))
    _, arg_s = _time_call(arg_grad, z, iters=iters)
    _, param_s = _time_call(param_grad, a, iters=iters)
    return {"argument_grad_s": round(arg_s, 6), "parameter_grad_s": round(param_s, 6)}


def _bench_barnes(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import double_gamma

    tau = jnp.float64(1.0)
    z = jnp.asarray(1.1 + 0.05j, dtype=jnp.complex128)
    import jax

    arg_grad = jax.jacfwd(lambda xv: jnp.real(double_gamma.ifj_barnesdoublegamma(jnp.asarray(xv + 0.05j, dtype=jnp.complex128), tau, dps=60)))
    param_grad = jax.jacfwd(lambda tv: jnp.real(double_gamma.ifj_barnesdoublegamma(z, tv, dps=60)))
    _, arg_s = _time_call(arg_grad, jnp.float64(1.1), iters=iters)
    _, param_s = _time_call(param_grad, tau, iters=iters)
    return {"argument_grad_s": round(arg_s, 6), "parameter_grad_s": round(param_s, 6)}


def _bench_bessel(iters: int) -> dict[str, float]:
    import jax.numpy as jnp
    from arbplusjax import bessel_kernels as bk

    nu = jnp.float32(0.4)
    z = jnp.float32(2.5)
    import jax

    arg_grad = jax.grad(lambda zv: bk.real_bessel_eval_j(nu, zv))
    param_grad = jax.grad(lambda nv: bk.real_bessel_eval_j(nv, z))
    _, arg_s = _time_call(arg_grad, z, iters=iters)
    _, param_s = _time_call(param_grad, nu, iters=iters)
    return {"argument_grad_s": round(arg_s, 6), "parameter_grad_s": round(param_s, 6)}


def _run_family(name: str, iters: int) -> dict[str, float]:
    if name == "incomplete_gamma_upper":
        return _bench_gamma(iters)
    if name == "hypgeom_1f1":
        return _bench_hypgeom_1f1(iters)
    if name == "hypgeom_u":
        return _bench_hypgeom_u(iters)
    if name == "barnesdoublegamma_ifj":
        return _bench_barnes(iters)
    if name == "bessel_j":
        return _bench_bessel(iters)
    raise ValueError(f"unknown family: {name}")


def _run_family_subprocess(name: str, iters: int) -> dict[str, float]:
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--family",
            name,
            "--iters",
            str(iters),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark argument-direction and parameter-direction AD for representative special-function families.")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--family", choices=FAMILY_RUNNERS)
    parser.add_argument(
        "--out-json",
        default="benchmarks/results/special_function_ad_benchmark/special_function_ad_benchmark.json",
    )
    parser.add_argument(
        "--out-md",
        default="benchmarks/results/special_function_ad_benchmark/special_function_ad_benchmark.md",
    )
    args = parser.parse_args()

    if args.family is not None:
        print(json.dumps(_run_family(args.family, args.iters)))
        return

    payload = {name: _run_family_subprocess(name, args.iters) for name in FAMILY_RUNNERS}

    out_json = REPO_ROOT / args.out_json
    out_md = REPO_ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Last updated: 2026-03-26T00:00:00Z",
        "",
        "# Special-Function AD Benchmark",
        "",
        "Generated by `benchmarks/special_function_ad_benchmark.py`.",
        "",
    ]
    for name, values in payload.items():
        lines.extend(
            [
                f"## {name}",
                f"- argument grad s: `{values['argument_grad_s']}`",
                f"- parameter grad s: `{values['parameter_grad_s']}`",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
