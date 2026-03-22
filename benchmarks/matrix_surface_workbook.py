from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from _source_tree_bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # package import path
    from benchmarks._source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)

from benchmarks import benchmark_dense_matrix_surface as dense_bench
from benchmarks import benchmark_matrix_backend_candidates as candidate_bench
from benchmarks import benchmark_matrix_free_krylov as mf_bench
from benchmarks import benchmark_sparse_matrix_surface as sparse_bench


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "matrix_surface_workbook.md"


@dataclass(frozen=True)
class Section:
    title: str
    command: str
    results: dict[str, float]


def _format_table(results: dict[str, float]) -> str:
    lines = [
        "| metric | seconds |",
        "| --- | ---: |",
    ]
    for key in sorted(results):
        lines.append(f"| `{key}` | {results[key]:.6e} |")
    return "\n".join(lines)


def _sections(n: int, warmup: int, runs: int, steps: int) -> list[Section]:
    return [
        Section(
            title="Dense Matrix Surface",
            command=f"python benchmarks/benchmark_dense_matrix_surface.py --n {n} --warmup {warmup} --runs {runs}",
            results={**dense_bench.run_arb_case(n, warmup, runs), **dense_bench.run_acb_case(n, warmup, runs)},
        ),
        Section(
            title="Sparse Matrix Surface",
            command=f"python benchmarks/benchmark_sparse_matrix_surface.py --n {n} --warmup {warmup} --runs {runs}",
            results={**sparse_bench.run_srb_case(n, warmup, runs), **sparse_bench.run_scb_case(n, warmup, runs)},
        ),
        Section(
            title="Matrix-Free Surface",
            command="python benchmarks/benchmark_matrix_free_krylov.py",
            results={
                **mf_bench.run_real_case(n=n, steps=steps),
                **mf_bench.run_sparse_real_parametric_case(steps=steps),
                **mf_bench.run_complex_case(n=max(2, n // 2), steps=max(2, steps - 2)),
                **mf_bench.run_sparse_complex_case(steps=max(2, steps - 4)),
            },
        ),
        Section(
            title="Matrix Backend Candidates",
            command=f"python benchmarks/benchmark_matrix_backend_candidates.py --n {n} --warmup {warmup} --runs {runs}",
            results=candidate_bench.run_candidate_suite(n=n, warmup=warmup, runs=runs, density=0.15),
        ),
    ]


def render(n: int = 8, warmup: int = 1, runs: int = 2, steps: int = 6) -> str:
    sections = _sections(n, warmup, runs, steps)
    lines = [
        "Last updated: 2026-03-20T00:00:00Z",
        "",
        "# Matrix Surface Workbook",
        "",
        "This workbook summarizes the benchmark surfaces for dense, sparse, and matrix-free matrix families.",
        "",
        "It is intended to make the current comparison surface legible while the matrix API and execution-strategy hardening continues.",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {section.title}",
                "",
                "Command:",
                "",
                "```bash",
                section.command,
                "```",
                "",
                _format_table(section.results),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the matrix surface workbook report.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    text = render(n=args.n, warmup=args.warmup, runs=args.runs, steps=args.steps)
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
