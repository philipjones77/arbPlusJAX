from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> int:
    print(f"[{_ts()}] cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    cp = subprocess.run(cmd, cwd=str(cwd), env=env)
    return cp.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run example benchmark profiles from examples/_input JSON config.")
    parser.add_argument("--config", default="examples/_input/example_run.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / args.config).resolve()
    if not cfg_path.exists():
        print(f"Missing config: {cfg_path}")
        print("Copy examples/example_run_template.json to examples/_input/example_run.json and edit it.")
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    python_bin = cfg.get("python") or os.getenv("ARBPLUSJAX_PYTHON") or sys.executable
    jax_mode = cfg.get("jax_mode", "cpu")
    jax_dtype = cfg.get("jax_dtype", "float64")
    c_ref_dir = cfg.get("c_ref_dir", "")
    profiles = cfg.get("profiles", [])

    if not profiles:
        print("No profiles in config.")
        return 2

    out_root = repo_root / "examples" / "_output" / cfg.get("run_name", f"run_{time.strftime('%Y%m%dT%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ARBPLUSJAX_PYTHON"] = python_bin

    for idx, p in enumerate(profiles, start=1):
        name = p["name"]
        functions = p["functions"]
        samples = p.get("samples", "300,600")
        seeds = p.get("seeds", "0,1")
        prec_bits = str(p.get("prec_bits", 200))

        profile_out = out_root / name
        cmd = [
            python_bin,
            "tools/run_harness_profile.py",
            "--name",
            name,
            "--outdir",
            str(profile_out),
            "--functions",
            functions,
            "--samples",
            samples,
            "--seeds",
            seeds,
            "--jax-mode",
            jax_mode,
            "--jax-dtype",
            jax_dtype,
            "--prec-bits",
            prec_bits,
        ]
        if c_ref_dir:
            cmd.extend(["--c-ref-dir", c_ref_dir])

        print(f"[{_ts()}] profile {idx}/{len(profiles)}: {name}")
        rc = _run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)
        if rc != 0:
            print(f"[{_ts()}] failed profile: {name} rc={rc}")
            return rc

    print(f"[{_ts()}] suite complete: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
