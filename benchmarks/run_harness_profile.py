from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _run_with_peak_rss(cmd: list[str], env: dict[str, str], cwd: Path) -> tuple[int, float, float]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    peak = 0
    p = psutil.Process(proc.pid) if psutil is not None else None

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
        if p is not None:
            try:
                rss = p.memory_info().rss
                for ch in p.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                peak = max(peak, rss)
            except Exception:
                pass

    rc = proc.wait()
    wall_s = time.perf_counter() - t0
    peak_mb = (peak / (1024.0 * 1024.0)) if peak > 0 else 0.0
    return rc, wall_s, peak_mb


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def _to_float(x: str) -> float | None:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _build_markdown(rows: list[dict[str, str]]) -> str:
    by_backend: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)

    lines = ["# Profile Summary", "", "| backend | rows | mean_time_ms | mean_abs_err | mean_containment | max_peak_rss_mb |", "|---|---:|---:|---:|---:|---:|"]
    for backend in sorted(by_backend):
        rs = by_backend[backend]
        times = [_to_float(r.get("time_ms", "")) for r in rs]
        errs = [_to_float(r.get("mean_abs_err", "")) for r in rs]
        cont = [_to_float(r.get("containment_rate", "")) for r in rs]
        rss = [_to_float(r.get("peak_rss_mb", "")) for r in rs]

        def _mean(vals: list[float | None]) -> float:
            f = [v for v in vals if v is not None]
            return sum(f) / len(f) if f else 0.0

        max_rss = max([v for v in rss if v is not None], default=0.0)
        lines.append(
            f"| {backend} | {len(rs)} | {_mean(times):.6g} | {_mean(errs):.6g} | {_mean(cont):.6g} | {max_rss:.6g} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bench_harness sweeps and collect time/memory/accuracy summaries.")
    parser.add_argument("--functions", required=True, help="Comma-separated benchmark functions.")
    parser.add_argument("--samples", default="300,600", help="Comma-separated sample counts.")
    parser.add_argument("--seeds", default="0,1", help="Comma-separated seeds.")
    parser.add_argument("--jax-mode", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--c-ref-dir", default=os.getenv("ARB_C_REF_DIR", ""))
    parser.add_argument("--prec-bits", type=int, default=200)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--jax-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--jax-fixed-batch-size",
        type=int,
        default=0,
        help="Fixed leading batch size for JAX kernels. Default (0) uses max(samples) for stable shapes across sweep.",
    )
    parser.add_argument(
        "--no-jax-warmup",
        action="store_true",
        help="Disable JAX warmup. By default warmup is enabled so timings focus on steady-state execution.",
    )
    parser.add_argument(
        "--split-process",
        action="store_true",
        help="Run one bench_harness process per (sample, seed). Default uses a single process for the full sweep to improve JAX cache reuse.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = os.getenv("ARBPLUSJAX_PYTHON", sys.executable)

    samples = _parse_int_list(args.samples)
    seeds = _parse_int_list(args.seeds)
    if not samples or not seeds:
        print("Need non-empty --samples and --seeds", file=sys.stderr)
        return 2

    run_name = args.name.strip() or f"profile_{_now_tag()}"
    base_out = Path(args.outdir) if args.outdir else (repo_root / "benchmarks" / "results" / run_name)
    base_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Persist compilation artifacts across runs so JAX can reuse compiled executables.
    if not env.get("JAX_COMPILATION_CACHE_DIR"):
        cache_dir = repo_root / "experiments" / "benchmarks" / "outputs" / "cache" / "jax_compilation_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    if args.jax_mode == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif args.jax_mode == "gpu":
        env["JAX_PLATFORMS"] = "cuda"

    fixed_batch = args.jax_fixed_batch_size if args.jax_fixed_batch_size > 0 else max(samples)
    use_warmup = not args.no_jax_warmup

    all_rows: list[dict[str, str]] = []
    if args.split_process:
        for s in samples:
            for seed in seeds:
                out_dir = base_out / f"samples_{s}_seed_{seed}"
                cmd = [
                    py,
                    "benchmarks/bench_harness.py",
                    "--samples",
                    str(s),
                    "--seed",
                    str(seed),
                    "--sweep-samples",
                    str(s),
                    "--sweep-seeds",
                    str(seed),
                    "--functions",
                    args.functions,
                    "--prec-bits",
                    str(args.prec_bits),
                    "--dps",
                    str(args.dps),
                    "--jax-dtype",
                    args.jax_dtype,
                    "--jax-batch",
                    "--jax-point-batch",
                    "--jax-fixed-batch-size",
                    str(fixed_batch),
                    "--outdir",
                    str(out_dir),
                ]
                if use_warmup:
                    cmd.append("--jax-warmup")
                if args.c_ref_dir:
                    cmd.extend(["--c-ref-dir", args.c_ref_dir])

                print(f"[run] samples={s} seed={seed} functions={args.functions}")
                rc, wall_s, peak_mb = _run_with_peak_rss(cmd, env=env, cwd=repo_root)
                if rc != 0:
                    print(f"Run failed for samples={s} seed={seed} rc={rc}", file=sys.stderr)
                    return rc

                summary_path = out_dir / f"samples_{s}_seed_{seed}" / "summary.csv"
                rows = _read_summary_csv(summary_path)
                for r in rows:
                    r["sweep_sample"] = str(s)
                    r["sweep_seed"] = str(seed)
                    r["wall_s"] = f"{wall_s:.6f}"
                    r["peak_rss_mb"] = f"{peak_mb:.6f}"
                    r["run_dir"] = str(summary_path.parent)
                    all_rows.append(r)
    else:
        sweep_samples = ",".join(str(x) for x in samples)
        sweep_seeds = ",".join(str(x) for x in seeds)
        cmd = [
            py,
            "benchmarks/bench_harness.py",
            "--samples",
            str(samples[0]),
            "--seed",
            str(seeds[0]),
            "--sweep-samples",
            sweep_samples,
            "--sweep-seeds",
            sweep_seeds,
            "--functions",
            args.functions,
            "--prec-bits",
            str(args.prec_bits),
            "--dps",
            str(args.dps),
            "--jax-dtype",
            args.jax_dtype,
            "--jax-batch",
            "--jax-point-batch",
            "--jax-fixed-batch-size",
            str(fixed_batch),
            "--outdir",
            str(base_out),
        ]
        if use_warmup:
            cmd.append("--jax-warmup")
        if args.c_ref_dir:
            cmd.extend(["--c-ref-dir", args.c_ref_dir])

        print(f"[run] sweep samples={sweep_samples} seeds={sweep_seeds} functions={args.functions}")
        wall_sweep, peak_sweep = 0.0, 0.0
        rc, wall_sweep, peak_sweep = _run_with_peak_rss(cmd, env=env, cwd=repo_root)
        if rc != 0:
            print(f"Run failed for sweep samples={sweep_samples} seeds={sweep_seeds} rc={rc}", file=sys.stderr)
            return rc

        sweep_index = base_out / "sweep_index.json"
        if not sweep_index.exists():
            print(f"Missing sweep index: {sweep_index}", file=sys.stderr)
            return 4

        runs = json.loads(sweep_index.read_text(encoding="utf-8"))
        for run in runs:
            run_dir = Path(run["path"])
            s = int(run["samples"])
            seed = int(run["seed"])
            summary_path = run_dir / "summary.csv"
            rows = _read_summary_csv(summary_path)
            for r in rows:
                r["sweep_sample"] = str(s)
                r["sweep_seed"] = str(seed)
                r["wall_s"] = f"{wall_sweep:.6f}"
                r["peak_rss_mb"] = f"{peak_sweep:.6f}"
                r["run_dir"] = str(run_dir)
                all_rows.append(r)

    if not all_rows:
        print("No summary rows were collected.", file=sys.stderr)
        return 3

    headers = list(all_rows[0].keys())
    csv_out = base_out / "profile_summary.csv"
    _write_csv(csv_out, all_rows, headers)

    md_out = base_out / "profile_summary.md"
    md_out.write_text(_build_markdown(all_rows), encoding="utf-8")

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
