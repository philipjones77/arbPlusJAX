from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.reference_backends import apply_reference_env
from benchmarks.reference_backends import default_boost_ref_cmd


def _default_c_ref_dir(repo_root: Path) -> str:
    env = os.getenv("ARB_C_REF_DIR", "")
    if env:
        return env
    candidates = [
        repo_root.parent / "flint" / "build",
        repo_root.parent / "arb" / "build",
        repo_root / "stuff" / "migration" / "c_chassis" / "build_linux_wsl",
        repo_root / "stuff" / "migration" / "c_chassis" / "build",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def _apply_jax_runtime_env(repo_root: Path, env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    cache_dir = (
        Path(out.get("JAX_COMPILATION_CACHE_DIR", "")).expanduser()
        if out.get("JAX_COMPILATION_CACHE_DIR")
        else (repo_root / "experiments" / "benchmarks" / "outputs" / "cache" / "jax_compilation_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    out.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    out.setdefault("JAX_ENABLE_X64", "1")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run arbPlusJAX benchmark sweeps with optional external baselines."
    )
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--c-ref-dir", type=str, default="")
    parser.add_argument("--functions", type=str, default="", help="Comma-separated function subset.")
    parser.add_argument("--samples", type=int, default=0, help="Override samples per sweep.")
    parser.add_argument("--sweep-samples", type=str, default="", help="Override sweep samples list.")
    parser.add_argument("--sweep-seeds", type=str, default="", help="Override sweep seeds list.")
    parser.add_argument("--with-mathematica", action="store_true")
    parser.add_argument("--with-boost", action="store_true")
    parser.add_argument(
        "--no-jax-batch",
        action="store_true",
        help="Disable batched JAX interval evaluation in harness runs.",
    )
    parser.add_argument(
        "--boost-ref-cmd",
        type=str,
        default=os.getenv("BOOST_REF_CMD", ""),
        help="Optional Boost reference command; reads JSON stdin and prints JSON array stdout.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    apply_reference_env(repo_root)
    harness = repo_root / "benchmarks" / "bench_harness.py"
    cmd = [sys.executable, str(harness), "--seed", str(args.seed)]
    env = _apply_jax_runtime_env(repo_root, os.environ.copy())

    c_ref_dir = args.c_ref_dir or _default_c_ref_dir(repo_root)
    if c_ref_dir:
        cmd.extend(["--c-ref-dir", c_ref_dir])
    else:
        print("Warning: no C/FLINT reference build found; C baseline will be skipped.")

    if args.profile == "quick":
        cmd.extend(["--samples", "200", "--sweep-samples", "200", "--sweep-seeds", str(args.seed)])
    else:
        cmd.extend(["--samples", "5000", "--sweep-samples", "2000,5000", "--sweep-seeds", "0,1"])
    if args.samples > 0:
        cmd.extend(["--samples", str(args.samples)])
    if args.sweep_samples:
        cmd.extend(["--sweep-samples", args.sweep_samples])
    if args.sweep_seeds:
        cmd.extend(["--sweep-seeds", args.sweep_seeds])
    if args.functions:
        cmd.extend(["--functions", args.functions])
    if not args.no_jax_batch:
        cmd.append("--jax-batch")

    if args.outdir:
        cmd.extend(["--outdir", args.outdir])
    if args.with_mathematica:
        cloud = os.getenv("WOLFRAM_CLOUD_URL", "")
        if cloud:
            cmd.extend(["--wolfram-cloud-url", cloud])
    if args.with_boost:
        boost_ref_cmd = args.boost_ref_cmd or default_boost_ref_cmd(repo_root)
        if boost_ref_cmd:
            cmd.extend(["--boost-ref-cmd", boost_ref_cmd])
        else:
            print("Warning: --with-boost was set but no boost command was provided (set --boost-ref-cmd or BOOST_REF_CMD).")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
