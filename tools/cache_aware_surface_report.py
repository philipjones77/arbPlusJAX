from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "cache_aware_surface_inventory.md"
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_api():
    from arbplusjax import api

    return api


def _cache_surface_rows() -> list[tuple[str, str, str]]:
    api = _load_api()
    rows: list[tuple[str, str, str]] = []
    for entry in api.list_public_function_metadata():
        name = entry.name
        if "cached_" in name:
            if "prepare" in name:
                reuse = "prepared-plan build"
            elif "apply" in name:
                reuse = "prepared-plan apply"
            else:
                reuse = "cache-aware helper"
            rows.append((name, entry.family, reuse))
    return sorted(rows, key=lambda row: row[0])


def _binder_rows() -> list[tuple[str, str]]:
    return [
        ("api.bind_point_batch", "bind reusable point-batch callable"),
        ("api.bind_point_batch_jit", "bind reusable compiled point-batch callable"),
        ("api.bind_interval_batch", "bind reusable interval-batch callable"),
    ]


def _example_rows() -> list[tuple[str, str]]:
    return [
        ("examples/example_api_surface.ipynb", "bound API callable reuse and runtime parameterization"),
        ("examples/example_core_scalar_surface.ipynb", "bound point-batch reuse and stable dtype/mode policy"),
        ("examples/example_dense_matrix_surface.ipynb", "cached dense prepare/apply reuse"),
        ("examples/example_sparse_matrix_surface.ipynb", "cached sparse and block/vblock prepare/apply reuse"),
        ("examples/example_matrix_free_operator_surface.ipynb", "operator-plan and preconditioner reuse"),
        ("examples/example_fft_nufft_surface.ipynb", "prepared transform plan reuse"),
        ("examples/example_gamma_family_surface.ipynb", "bound callable reuse with stable point-mode controls"),
        ("examples/example_barnes_double_gamma_surface.ipynb", "bound callable reuse with stable special-function controls"),
    ]


def _benchmark_rows() -> list[tuple[str, str]]:
    return [
        ("benchmarks/matrix_surface_workbook.py", "dense, sparse, block, vblock, and matrix-free reuse comparisons"),
        ("benchmarks/run_hypgeom_benchmark_smoke.py", "fixed-shape padded hypgeom batch reuse"),
        ("benchmarks/benchmark_fft_nufft.py", "prepared transform plan reuse"),
        ("benchmarks/benchmark_sparse_matrix_surface.py", "cached sparse prepare/apply reuse"),
        ("benchmarks/benchmark_dense_matrix_surface.py", "dense repeated-call and cached apply behavior"),
    ]


def render() -> str:
    lines = [
        "Last updated: 2026-03-24T00:00:00Z",
        "",
        "# Cache-Aware Surface Inventory",
        "",
        "This generated report records the current cache-aware public surfaces and the canonical examples and benchmarks that should demonstrate compliant reuse patterns.",
        "",
        "Refresh with `python tools/cache_aware_surface_report.py` or the umbrella `python tools/check_generated_reports.py` path.",
        "",
        "## Bound API Reuse Surfaces",
        "",
        "| surface | purpose |",
        "|---|---|",
    ]
    for name, purpose in _binder_rows():
        lines.append(f"| `{name}` | {purpose} |")
    lines.extend(
        [
            "",
            "## Public Cached Prepare/Apply Surfaces",
            "",
            "| public name | family | reuse role |",
            "|---|---|---|",
        ]
    )
    for name, family, reuse in _cache_surface_rows():
        lines.append(f"| `{name}` | `{family}` | {reuse} |")
    lines.extend(
        [
            "",
            "## Canonical Example Evidence",
            "",
            "| example | required reuse pattern |",
            "|---|---|",
        ]
    )
    for path, purpose in _example_rows():
        lines.append(f"| [{path}](/{path}) | {purpose} |")
    lines.extend(
        [
            "",
            "## Canonical Benchmark Evidence",
            "",
            "| benchmark entrypoint | required reuse pattern |",
            "|---|---|",
        ]
    )
    for path, purpose in _benchmark_rows():
        lines.append(f"| [{path}](/{path}) | {purpose} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
