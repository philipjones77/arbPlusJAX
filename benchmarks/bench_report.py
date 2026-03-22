from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _pick_run(path: Path) -> Path:
    if path.is_file():
        return path.parent
    candidates = sorted(path.glob("samples_*_seed_*/summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No summary.csv found under {path}")
    return candidates[-1].parent


def _to_float(val: str) -> float | None:
    try:
        return float(val)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a concise benchmark markdown report.")
    parser.add_argument("--run", type=str, default="", help="Run dir or summary.csv path.")
    parser.add_argument("--out", type=str, default="", help="Output markdown path.")
    args = parser.parse_args()

    base = Path(args.run) if args.run else Path("benchmarks") / "results"
    run_dir = _pick_run(base)
    summary_csv = run_dir / "summary.csv"
    rows = list(csv.DictReader(summary_csv.read_text().splitlines()))
    backends = sorted({r["backend"] for r in rows})

    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- Run: `{run_dir}`")
    lines.append(f"- Rows: {len(rows)}")
    lines.append("")
    lines.append("## Backend Summary")
    lines.append("")
    lines.append("| Backend | Rows | Mean time (ms) | Mean abs err | Mean containment |")
    lines.append("|---|---:|---:|---:|---:|")
    for backend in backends:
        subset = [r for r in rows if r["backend"] == backend]
        tvals = [_to_float(r["time_ms"]) for r in subset]
        evals = [_to_float(r["mean_abs_err"]) for r in subset]
        cvals = [_to_float(r["containment_rate"]) for r in subset]
        tnums = [v for v in tvals if v is not None]
        enums = [v for v in evals if v is not None]
        cnums = [v for v in cvals if v is not None]
        tmean = sum(tnums) / len(tnums) if tnums else 0.0
        emean = sum(enums) / len(enums) if enums else 0.0
        cmean = sum(cnums) / len(cnums) if cnums else 0.0
        lines.append(f"| {backend} | {len(subset)} | {tmean:.6g} | {emean:.6g} | {cmean:.6g} |")

    summary_json = run_dir / "summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text())
        meta = payload.get("meta", {})
        lines.append("")
        lines.append("## Metadata")
        lines.append("")
        for key in ("samples", "seed", "dps", "prec_bits", "c_ref_dir", "boost_ref_cmd"):
            if key in meta:
                lines.append(f"- `{key}`: `{meta[key]}`")

    report = "\n".join(lines) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Wrote {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
