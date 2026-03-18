from __future__ import annotations

import argparse
import json
import os
import time


def _bool_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _backend_kind(name: str) -> str:
    low = name.strip().lower()
    if low == "cpu":
        return "cpu"
    if low in ("gpu", "cuda", "rocm"):
        return "gpu"
    if low == "tpu":
        return "tpu"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print JAX runtime/device info and optional quick benchmark.")
    parser.add_argument("--quick-bench", action="store_true", help="Run a small JIT matmul benchmark.")
    parser.add_argument("--size", type=int, default=2048, help="Matrix size for --quick-bench.")
    parser.add_argument(
        "--expect-backend",
        choices=("cpu", "gpu", "tpu"),
        default="",
        help="Fail with non-zero exit code if the backend kind does not match.",
    )
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from arbplusjax import precision

    precision.enable_jax_x64()

    payload = {
        "platform": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "env": {
            "JAX_PLATFORM_NAME": _bool_env("JAX_PLATFORM_NAME"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": _bool_env("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": _bool_env("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "XLA_FLAGS": _bool_env("XLA_FLAGS"),
        },
    }

    if args.quick_bench:
        n = int(args.size)
        x = jnp.ones((n, n), dtype=jnp.float64)
        y = jnp.ones((n, n), dtype=jnp.float64)

        @jax.jit
        def matmul(a, b):
            return a @ b

        t0 = time.perf_counter()
        out = matmul(x, y)
        out.block_until_ready()
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        out = matmul(x, y)
        out.block_until_ready()
        t3 = time.perf_counter()

        payload["quick_bench"] = {
            "size": n,
            "compile_plus_first_run_s": round(t1 - t0, 6),
            "steady_state_run_s": round(t3 - t2, 6),
        }

    print(json.dumps(payload, indent=2))

    if args.expect_backend:
        got = _backend_kind(str(payload["platform"]))
        if got != args.expect_backend:
            print(
                f"Expected backend kind '{args.expect_backend}' but got '{got}' (platform='{payload['platform']}').",
                flush=True,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
