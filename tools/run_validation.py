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

from python_resolver import jax_platform_env
from python_resolver import resolve_python


@dataclass
class Task:
    name: str
    cmd: list[str]
    env_overrides: dict[str, str]


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _stream_task(task: Task, cwd: Path, heartbeat_s: int) -> int:
    env = os.environ.copy()
    env.update(task.env_overrides)

    cmd_pretty = " ".join(task.cmd)
    print(f"[{_ts()}] START {task.name}: {cmd_pretty}", flush=True)

    proc = subprocess.Popen(
        task.cmd,
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

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    start = time.time()
    while True:
        try:
            line = q.get(timeout=heartbeat_s)
            if line is None:
                break
            print(f"[{_ts()}] {task.name}: {line}", flush=True)
        except queue.Empty:
            elapsed = int(time.time() - start)
            print(f"[{_ts()}] {task.name}: still running ({elapsed}s elapsed)", flush=True)

    rc = proc.wait()
    elapsed = int(time.time() - start)
    if rc == 0:
        print(f"[{_ts()}] DONE {task.name}: success ({elapsed}s)", flush=True)
    else:
        print(f"[{_ts()}] DONE {task.name}: failed rc={rc} ({elapsed}s)", flush=True)
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tests/benchmarks with live status on Linux or Windows.")
    parser.add_argument("--python", default="", help="Python interpreter to use. Default: auto-detect envs/jax.")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto", help="Set JAX_PLATFORMS.")
    parser.add_argument("--heartbeat", type=int, default=20, help="Heartbeat interval seconds.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest tests/.")
    parser.add_argument("--skip-benchmark-smoke", action="store_true", help="Skip pytest benchmark marker smoke.")
    parser.add_argument(
        "--benchmark-profile",
        choices=("none", "quick", "full"),
        default="quick",
        help="Run tools/run_benchmarks.py profile after smoke.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = resolve_python(args.python or None)
    py_env = os.environ.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    if py_env:
        py_env = src_path + os.pathsep + py_env
    else:
        py_env = src_path
    common_env = {"PYTHONPATH": py_env, **jax_platform_env(args.jax_mode)}
    print(f"[{_ts()}] Using Python: {py}", flush=True)
    print(f"[{_ts()}] JAX mode: {args.jax_mode}", flush=True)

    tasks: list[Task] = []
    if not args.skip_tests:
        tasks.append(
            Task(
                name="tests",
                cmd=[py, "-m", "pytest", "-ra", "-q", "tests"],
                env_overrides=common_env,
            )
        )
    if not args.skip_benchmark_smoke:
        tasks.append(
            Task(
                name="benchmark-smoke",
                cmd=[py, "-m", "pytest", "-ra", "-q", "benchmarks", "-m", "benchmark"],
                env_overrides={**common_env, "ARBPLUSJAX_RUN_BENCHMARKS": "1"},
            )
        )
    if args.benchmark_profile != "none":
        tasks.append(
            Task(
                name=f"benchmark-{args.benchmark_profile}",
                cmd=[py, "tools/run_benchmarks.py", "--profile", args.benchmark_profile],
                env_overrides=common_env,
            )
        )

    if not tasks:
        print("No tasks selected.")
        return 0

    failures = []
    for task in tasks:
        rc = _stream_task(task, repo_root, args.heartbeat)
        if rc != 0:
            failures.append((task.name, rc))
            break

    if failures:
        name, rc = failures[0]
        print(f"[{_ts()}] Validation stopped at {name} (rc={rc}).", flush=True)
        return rc

    print(f"[{_ts()}] Validation complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
