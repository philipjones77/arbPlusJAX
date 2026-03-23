from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import jax

try:
    from _source_tree_bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # package import path
    from benchmarks._source_tree_bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)

from benchmarks import benchmark_dense_matrix_surface as dense_bench
from benchmarks import benchmark_block_sparse_matrix_surface as block_sparse_bench
from benchmarks import benchmark_matrix_backend_candidates as candidate_bench
from benchmarks import benchmark_matrix_free_krylov as mf_bench
from benchmarks import benchmark_sparse_matrix_surface as sparse_bench
from benchmarks import benchmark_vblock_sparse_matrix_surface as vblock_sparse_bench
import jax.numpy as jnp


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


def _top_metrics_line(results: dict[str, float], *, limit: int = 4) -> str:
    top = sorted(results.items(), key=lambda item: item[1])[:limit]
    return ", ".join(f"`{name}`={value:.3e}s" for name, value in top)


def _compare_table(sections: list[Section]) -> str:
    lines = [
        "| family | use when | fastest metrics |",
        "| --- | --- | --- |",
        "| dense | matrices are small/medium and direct kernels or cached dense plans are appropriate | "
        + _top_metrics_line(next(section.results for section in sections if section.title == "Dense Matrix Surface"))
        + " |",
        "| sparse | storage sparsity is meaningful and callers want sparse cached matvec/rmatvec reuse | "
        + _top_metrics_line(next(section.results for section in sections if section.title == "Sparse Matrix Surface"))
        + " |",
        "| block sparse | block structure is explicit and callers want block-native apply paths | "
        + _top_metrics_line(next(section.results for section in sections if section.title == "Block Sparse Matrix Surface"))
        + " |",
        "| variable block sparse | partitions are irregular but structure should still be preserved | "
        + _top_metrics_line(next(section.results for section in sections if section.title == "Variable-Block Sparse Matrix Surface"))
        + " |",
        "| matrix free | operator plans, Krylov solves, logdet, or adapter-based execution matter more than explicit materialization | "
        + _top_metrics_line(next(section.results for section in sections if section.title == "Matrix-Free Surface"))
        + " |",
    ]
    return "\n".join(lines)


def _sections(n: int, warmup: int, runs: int, steps: int) -> list[Section]:
    block_n = max(2, n // 2)
    block_real = block_sparse_bench._real_case(block_n, 2, jnp.float64)
    block_complex = block_sparse_bench._complex_case(block_n, 2, jnp.float64, jnp.complex128)
    vblock_real = vblock_sparse_bench._real_case(n, jnp.float64)
    vblock_complex = vblock_sparse_bench._complex_case(n, jnp.float64, jnp.complex128)
    return [
        Section(
            title="Dense Matrix Surface",
            command=f"python benchmarks/benchmark_dense_matrix_surface.py --n {n} --warmup {warmup} --runs {runs} --dtype float64 --smoke",
            results={
                **dense_bench.run_arb_case(n, warmup, runs, jnp.float64, smoke=True),
                **dense_bench.run_acb_case(n, warmup, runs, jnp.float64, jnp.complex128, smoke=True),
            },
        ),
        Section(
            title="Sparse Matrix Surface",
            command=f"python benchmarks/benchmark_sparse_matrix_surface.py --n {n} --warmup {warmup} --runs {runs} --dtype float64 --smoke",
            results={
                **sparse_bench.run_srb_case(n, warmup, runs, jnp.float64, smoke=True),
                **sparse_bench.run_scb_case(n, warmup, runs, jnp.float64, jnp.complex128, smoke=True),
            },
        ),
        Section(
            title="Block Sparse Matrix Surface",
            command=f"python benchmarks/benchmark_block_sparse_matrix_surface.py --n-blocks {block_n} --block-size 2 --warmup {warmup} --runs {runs} --dtype float64 --smoke",
            results={
                "srb_block_matvec_s": block_sparse_bench._time_call(jax.jit(block_sparse_bench.srb_block_mat.srb_block_mat_matvec), block_real[0], block_real[1], warmup=warmup, runs=runs),
                "srb_block_rmatvec_cached_s": block_sparse_bench._time_call(jax.jit(block_sparse_bench.srb_block_mat.srb_block_mat_rmatvec_cached_apply), block_sparse_bench.srb_block_mat.srb_block_mat_rmatvec_cached_prepare(block_real[0]), block_real[1], warmup=warmup, runs=runs),
                "scb_block_matvec_s": block_sparse_bench._time_call(jax.jit(block_sparse_bench.scb_block_mat.scb_block_mat_matvec), block_complex[0], block_complex[1], warmup=warmup, runs=runs),
                "scb_block_adjoint_cached_s": block_sparse_bench._time_call(jax.jit(block_sparse_bench.scb_block_mat.scb_block_mat_adjoint_matvec_cached_apply), block_sparse_bench.scb_block_mat.scb_block_mat_adjoint_matvec_cached_prepare(block_complex[0]), block_complex[1], warmup=warmup, runs=runs),
            },
        ),
        Section(
            title="Variable-Block Sparse Matrix Surface",
            command=f"python benchmarks/benchmark_vblock_sparse_matrix_surface.py --n {n} --warmup {warmup} --runs {runs} --dtype float64 --smoke",
            results={
                "srb_vblock_matvec_s": vblock_sparse_bench._time_call(jax.jit(vblock_sparse_bench.srb_vblock_mat.srb_vblock_mat_matvec), vblock_real[0], vblock_real[2], warmup=warmup, runs=runs),
                "srb_vblock_matvec_cached_s": vblock_sparse_bench._time_call(jax.jit(vblock_sparse_bench.srb_vblock_mat.srb_vblock_mat_matvec_cached_apply), vblock_real[1], vblock_real[2], warmup=warmup, runs=runs),
                "scb_vblock_matvec_s": vblock_sparse_bench._time_call(jax.jit(vblock_sparse_bench.scb_vblock_mat.scb_vblock_mat_matvec), vblock_complex[0], vblock_complex[2], warmup=warmup, runs=runs),
                "scb_vblock_matvec_cached_s": vblock_sparse_bench._time_call(jax.jit(vblock_sparse_bench.scb_vblock_mat.scb_vblock_mat_matvec_cached_apply), vblock_complex[1], vblock_complex[2], warmup=warmup, runs=runs),
            },
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
        "The workbook now compares dense, sparse, block-sparse, variable-block sparse, and matrix-free/operator-plan execution in one place.",
        "",
        "## Compare and Contrast",
        "",
        _compare_table(sections),
        "",
        "Recommended visualizations in the canonical notebooks:",
        "",
        "- dense: direct solve vs cached matvec/rmatvec vs operator-plan apply",
        "- sparse: sparse vs block-sparse vs vblock matvec/rmatvec",
        "- matrix-free: dense-adapted vs sparse-adapted operator plans and solve/logdet slices",
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
