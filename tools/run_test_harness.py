from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from python_resolver import jax_platform_env
from python_resolver import resolve_python
from runtime_manifest import collect_runtime_manifest
from runtime_manifest import write_runtime_manifest


@dataclass(frozen=True)
class TestTask:
    name: str
    args: list[str]
    env_overrides: dict[str, str]


def _run_task(py: str, repo_root: Path, task: TestTask) -> int:
    env = os.environ.copy()
    env.update(task.env_overrides)
    cmd = [py, "-m", "pytest", "-ra", "-q", *task.args]
    print(f"[test-harness] {task.name}: {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(repo_root), env=env)
    return int(completed.returncode)


def _common_env(repo_root: Path, jax_mode: str) -> dict[str, str]:
    current = os.environ.get("PYTHONPATH", "")
    src_path = str(repo_root / "src")
    py_path = src_path if not current else src_path + os.pathsep + current
    return {"PYTHONPATH": py_path, **jax_platform_env(jax_mode)}


def _build_tasks(args: argparse.Namespace, repo_root: Path) -> list[TestTask]:
    common_env = _common_env(repo_root, args.jax_mode)
    tasks: list[TestTask] = []

    if args.profile == "smoke":
        tasks.append(TestTask("smoke", ["tests/test_all_functions_smoke.py"], common_env))
    elif args.profile == "matrix":
        tasks.append(
            TestTask(
                "matrix",
                [
                    "tests/test_arb_mat_chassis.py",
                    "tests/test_acb_mat_chassis.py",
                    "tests/test_jrb_mat_chassis.py",
                    "tests/test_jcb_mat_chassis.py",
                    "tests/test_mat_modes.py",
                ],
                common_env,
            )
        )
    elif args.profile == "matrix-free":
        tasks.append(
            TestTask(
                "matrix-free",
                [
                    "tests/test_jrb_mat_chassis.py",
                    "tests/test_jcb_mat_chassis.py",
                ],
                common_env,
            )
        )
    elif args.profile == "special":
        tasks.append(
            TestTask(
                "special",
                [
                    "tests/test_tail_acceleration_scaffold.py",
                    "tests/test_incomplete_bessel_k.py",
                    "tests/test_incomplete_bessel_i.py",
                    "tests/test_incomplete_gamma.py",
                    "tests/test_laplace_bessel_k_tail.py",
                    "tests/test_api_metadata.py",
                ],
                common_env,
            )
        )
    elif args.profile == "chassis":
        tasks.append(TestTask("chassis", ["tests", "-m", "not parity"], common_env))
    elif args.profile == "parity":
        tasks.append(
            TestTask(
                "parity",
                ["tests", "-m", "parity"],
                {**common_env, "ARBPLUSJAX_RUN_PARITY": "1"},
            )
        )
    elif args.profile == "bench-smoke":
        tasks.append(
            TestTask(
                "bench-smoke",
                ["benchmarks", "-m", "benchmark"],
                {**common_env, "ARBPLUSJAX_RUN_BENCHMARKS": "1"},
            )
        )
    elif args.profile == "full":
        tasks.append(TestTask("chassis", ["tests", "-m", "not parity"], common_env))
        if args.with_bench_smoke:
            tasks.append(
                TestTask(
                    "bench-smoke",
                    ["benchmarks", "-m", "benchmark"],
                    {**common_env, "ARBPLUSJAX_RUN_BENCHMARKS": "1"},
                )
            )
        if args.with_parity:
            tasks.append(
                TestTask(
                    "parity",
                    ["tests", "-m", "parity"],
                    {**common_env, "ARBPLUSJAX_RUN_PARITY": "1"},
                )
            )
    else:
        raise ValueError(f"Unsupported profile: {args.profile}")

    if args.pytest_args:
        extra = args.pytest_args.split()
        tasks = [TestTask(task.name, [*task.args, *extra], task.env_overrides) for task in tasks]
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run arbPlusJAX tests with explicit environment-aware profiles.")
    parser.add_argument(
        "--profile",
        choices=("smoke", "matrix", "matrix-free", "special", "chassis", "parity", "bench-smoke", "full"),
        default="chassis",
        help="Named test profile to run.",
    )
    parser.add_argument("--python", default="", help="Python interpreter to use. Default: auto-detect envs/jax.")
    parser.add_argument("--jax-mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--with-parity", action="store_true", help="Only used by --profile full.")
    parser.add_argument("--with-bench-smoke", action="store_true", help="Only used by --profile full.")
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest arguments appended to every task, for example: --pytest-args \"-k matrix\"",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="Optional output directory. When set, writes runtime_manifest.json for the test run.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = resolve_python(args.python or None)
    print(f"[test-harness] python={py}", flush=True)
    print(f"[test-harness] jax_mode={args.jax_mode}", flush=True)
    print(f"[test-harness] profile={args.profile}", flush=True)
    if args.outdir:
        manifest = collect_runtime_manifest(repo_root, jax_mode=args.jax_mode, python_path=py)
        path = write_runtime_manifest(Path(args.outdir), manifest)
        print(f"[test-harness] wrote manifest {path}", flush=True)

    tasks = _build_tasks(args, repo_root)
    for task in tasks:
        rc = _run_task(py, repo_root, task)
        if rc != 0:
            print(f"[test-harness] stopping after {task.name} rc={rc}", flush=True)
            return rc

    print("[test-harness] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
