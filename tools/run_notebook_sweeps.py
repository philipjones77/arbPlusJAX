from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from benchmarks.bench_registry import FUNCTIONS
from python_resolver import jax_platform_env
from python_resolver import resolve_python


@dataclass
class SweepJob:
    sample: int
    seed: int
    functions: list[str]


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def _stream_cmd(cmd: list[str], cwd: Path, env: dict[str, str], label: str, heartbeat: int) -> int:
    print(f"[{_ts()}] START {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    q: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        for line in proc.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    start = time.time()
    while True:
        try:
            line = q.get(timeout=heartbeat)
            if line is None:
                break
            print(f"[{_ts()}] {label}: {line}", flush=True)
        except queue.Empty:
            print(f"[{_ts()}] {label}: still running ({int(time.time() - start)}s elapsed)", flush=True)

    rc = proc.wait()
    elapsed = int(time.time() - start)
    print(f"[{_ts()}] DONE {label}: rc={rc} ({elapsed}s)", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run larger benchmark sweeps with notebook-friendly progress logs.")
    parser.add_argument("--python", default="", help="Python interpreter to use. Default: auto-detect envs/jax.")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto", help="Set JAX_PLATFORMS.")
    parser.add_argument("--jax-dtype", choices=("float64", "float32"), default="float64", help="JAX dtype for JAX backends.")
    parser.add_argument("--samples", default="2000,5000,10000", help="Comma-separated sample sizes.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds.")
    parser.add_argument("--functions", default="", help="Optional comma-separated subset. Default: all registered functions.")
    parser.add_argument("--chunk-size", type=int, default=6, help="Functions per harness invocation.")
    parser.add_argument("--heartbeat", type=int, default=20, help="Heartbeat seconds.")
    parser.add_argument("--c-ref-dir", default=os.getenv("ARB_C_REF_DIR", ""), help="Optional C reference build path.")
    parser.add_argument("--outdir", default="", help="Optional output root for benchmark artifacts.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    py = resolve_python(args.python or None)

    if args.functions.strip():
        functions = [x.strip() for x in args.functions.split(",") if x.strip()]
    else:
        functions = [f.name for f in FUNCTIONS]

    samples = [int(x.strip()) for x in args.samples.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    function_chunks = _chunked(functions, args.chunk_size)
    jobs: list[SweepJob] = []
    for sample in samples:
        for seed in seeds:
            for fn_chunk in function_chunks:
                jobs.append(SweepJob(sample=sample, seed=seed, functions=fn_chunk))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(jax_platform_env(args.jax_mode))
    print(f"[{_ts()}] Using Python: {py}", flush=True)
    print(f"[{_ts()}] JAX mode: {args.jax_mode}", flush=True)
    print(f"[{_ts()}] JAX dtype: {args.jax_dtype}", flush=True)

    total = len(jobs)
    failures = 0
    for idx, job in enumerate(jobs, start=1):
        label = f"job {idx}/{total} sample={job.sample} seed={job.seed} fns={len(job.functions)}"
        cmd = [
            py,
            "benchmarks/bench_harness.py",
            "--seed",
            str(job.seed),
            "--samples",
            str(job.sample),
            "--sweep-samples",
            str(job.sample),
            "--sweep-seeds",
            str(job.seed),
            "--functions",
            ",".join(job.functions),
            "--jax-dtype",
            args.jax_dtype,
            "--jax-batch",
        ]
        if args.c_ref_dir:
            cmd.extend(["--c-ref-dir", args.c_ref_dir])
        if args.outdir:
            cmd.extend(["--outdir", args.outdir])

        print(f"[{_ts()}] QUEUE {label}: {job.functions}", flush=True)
        if args.dry_run:
            continue

        rc = _stream_cmd(cmd, repo_root, env, label, args.heartbeat)
        if rc != 0:
            failures += 1
            print(f"[{_ts()}] FAIL {label}", flush=True)

    if args.dry_run:
        print(f"[{_ts()}] Dry-run complete. Planned jobs: {total}", flush=True)
        return 0

    if failures:
        print(f"[{_ts()}] Sweep complete with {failures} failed job(s).", flush=True)
        return 1

    print(f"[{_ts()}] Sweep complete. Jobs run: {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
