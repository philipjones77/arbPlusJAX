from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.runtime_manifest import collect_runtime_manifest
from tools.runtime_manifest import write_runtime_manifest


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> int:
    print(f"[{_ts()}] cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    cp = subprocess.run(cmd, cwd=str(cwd), env=env)
    return cp.returncode


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _svg_bar_chart(title: str, items: list[tuple[str, float]], *, width: int = 960, height: int = 420) -> str:
    max_value = max((value for _, value in items), default=1.0)
    bar_width = max(40, int((width - 120) / max(len(items), 1)))
    baseline = height - 70
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f3eb"/>',
        f'<text x="40" y="36" font-size="24" font-family="Georgia, serif" fill="#1f2a30">{title}</text>',
        f'<line x1="40" y1="{baseline}" x2="{width - 40}" y2="{baseline}" stroke="#41535d" stroke-width="2"/>',
    ]
    for idx, (label, value) in enumerate(items):
        x = 60 + idx * bar_width
        bar_h = 0 if max_value <= 0 else int((value / max_value) * (height - 140))
        y = baseline - bar_h
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_width - 12}" height="{bar_h}" fill="#b85c38" rx="6"/>')
        lines.append(f'<text x="{x + 4}" y="{baseline + 20}" font-size="12" font-family="monospace" fill="#1f2a30">{label}</text>')
        lines.append(f'<text x="{x + 4}" y="{max(y - 8, 56)}" font-size="12" font-family="monospace" fill="#1f2a30">{value:.4g}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _write_suite_summary(out_root: Path, run_name: str, profile_dirs: list[Path], api_report_path: Path | None, diagnostics_path: Path | None) -> None:
    profile_rows: list[dict[str, str]] = []
    for profile_dir in profile_dirs:
        profile_rows.extend(_read_csv_rows(profile_dir / "profile_summary.csv"))

    backend_summary: dict[str, dict[str, list[float]]] = {}
    for row in profile_rows:
        backend = row.get("backend", "unknown")
        backend_summary.setdefault(backend, {"time_ms": [], "containment_rate": []})
        try:
            backend_summary[backend]["time_ms"].append(float(row.get("time_ms", "")))
        except Exception:
            pass
        try:
            backend_summary[backend]["containment_rate"].append(float(row.get("containment_rate", "")))
        except Exception:
            pass

    api_rows: list[dict[str, object]] = []
    if api_report_path is not None and api_report_path.exists():
        api_rows = json.loads(api_report_path.read_text(encoding="utf-8")).get("records", [])

    diag_rows: list[dict[str, object]] = []
    if diagnostics_path is not None and diagnostics_path.exists():
        diag_rows = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    lines = [
        f"# Example Run Suite Summary: {run_name}",
        "",
        "## Profile backends",
        "",
        "| backend | mean_time_ms | mean_containment | rows |",
        "|---|---:|---:|---:|",
    ]
    for backend in sorted(backend_summary):
        stats = backend_summary[backend]
        lines.append(
            f"| {backend} | {_mean(stats['time_ms']):.6g} | {_mean(stats['containment_rate']):.6g} | {len(stats['time_ms'])} |"
        )

    if api_rows:
        lines.extend(
            [
                "",
                "## API benchmark summary",
                "",
                "| operation | implementation | cold_time_s | warm_time_s | recompile_time_s |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in api_rows:
            lines.append(
                f"| {row['operation']} | {row['implementation']} | {row.get('cold_time_s', 0.0):.6g} | {row.get('warm_time_s', 0.0):.6g} | {row.get('recompile_time_s', 0.0):.6g} |"
            )

    if diag_rows:
        lines.extend(
            [
                "",
                "## Matrix diagnostics summary",
                "",
                "| case | compile_ms | steady_ms_median | recompile_new_shape_ms | peak_rss_delta_mb |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in diag_rows:
            lines.append(
                f"| {row['name']} | {row.get('compile_ms', 0.0):.6g} | {row.get('steady_ms_median', 0.0):.6g} | {row.get('recompile_new_shape_ms', 0.0):.6g} | {row.get('peak_rss_delta_mb', 0.0):.6g} |"
            )

    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    backend_items = [(name, _mean(stats["time_ms"])) for name, stats in sorted(backend_summary.items()) if stats["time_ms"]]
    if backend_items:
        (out_root / "profile_backend_time.svg").write_text(
            _svg_bar_chart("Profile Backend Mean Time (ms)", backend_items), encoding="utf-8"
        )

    api_items = [(f"{row['operation']}:{row['implementation']}", float(row.get("warm_time_s", 0.0))) for row in api_rows]
    if api_items:
        (out_root / "api_surface_warm_time.svg").write_text(
            _svg_bar_chart("API Warm Time (s)", api_items), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run example benchmark profiles from examples/inputs JSON config."
    )
    parser.add_argument(
        "--config",
        default="examples/inputs/example_run_suite/example_run.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / args.config).resolve()
    if not cfg_path.exists():
        print(f"Missing config: {cfg_path}")
        print(
            "Copy examples/inputs/example_run_suite/example_run_template.json "
            "to examples/inputs/example_run_suite/example_run.json and edit it."
        )
        return 2

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    python_bin = cfg.get("python") or os.getenv("ARBPLUSJAX_PYTHON") or sys.executable
    jax_mode = cfg.get("jax_mode", "cpu")
    jax_dtype = cfg.get("jax_dtype", "float64")
    c_ref_dir = cfg.get("c_ref_dir", "")
    profiles = cfg.get("profiles", [])
    api_benchmark = cfg.get("api_benchmark", {"enabled": True, "warmup": 1, "runs": 5})
    matrix_diagnostics = cfg.get("matrix_diagnostics", {"enabled": True, "n": 8, "repeats": 4})

    if not profiles:
        print("No profiles in config.")
        return 2

    run_name = cfg.get("run_name", f"run_{time.strftime('%Y%m%dT%H%M%S')}")
    out_root = repo_root / "examples" / "outputs" / "example_run_suite" / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ARBPLUSJAX_PYTHON"] = python_bin
    if jax_mode == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif jax_mode == "gpu":
        env["JAX_PLATFORMS"] = "cuda"
    env["JAX_ENABLE_X64"] = "1" if jax_dtype == "float64" else "0"

    runtime_manifest = collect_runtime_manifest(repo_root, jax_mode=jax_mode, python_path=python_bin)
    runtime_manifest["example_run_suite"] = {"jax_dtype": jax_dtype, "run_name": run_name}
    write_runtime_manifest(out_root, runtime_manifest)

    profile_dirs: list[Path] = []
    for idx, p in enumerate(profiles, start=1):
        name = p["name"]
        functions = p["functions"]
        samples = p.get("samples", "300,600")
        seeds = p.get("seeds", "0,1")
        prec_bits = str(p.get("prec_bits", 200))

        profile_out = out_root / name
        profile_dirs.append(profile_out)
        cmd = [
            python_bin,
            "benchmarks/run_harness_profile.py",
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

    api_report_path: Path | None = None
    if api_benchmark.get("enabled", True):
        api_report_path = out_root / "api_surface_report.json"
        cmd = [
            python_bin,
            "benchmarks/benchmark_api_surface.py",
            "--warmup",
            str(api_benchmark.get("warmup", 1)),
            "--runs",
            str(api_benchmark.get("runs", 5)),
            "--output",
            str(api_report_path),
        ]
        rc = _run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)
        if rc != 0:
            print(f"[{_ts()}] failed api benchmark rc={rc}")
            return rc

    diagnostics_path: Path | None = None
    if matrix_diagnostics.get("enabled", True):
        diagnostics_path = out_root / "matrix_stack_profile.json"
        cmd = [
            python_bin,
            "benchmarks/benchmark_matrix_stack_diagnostics.py",
            "--n",
            str(matrix_diagnostics.get("n", 8)),
            "--repeats",
            str(matrix_diagnostics.get("repeats", 4)),
            "--output",
            str(diagnostics_path),
        ]
        rc = _run(cmd, cwd=repo_root, env=env, dry_run=args.dry_run)
        if rc != 0:
            print(f"[{_ts()}] failed matrix diagnostics rc={rc}")
            return rc

    if not args.dry_run:
        _write_suite_summary(out_root, run_name, profile_dirs, api_report_path, diagnostics_path)

    print(f"[{_ts()}] suite complete: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
