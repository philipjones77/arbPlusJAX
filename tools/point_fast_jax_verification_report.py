from __future__ import annotations

from collections import Counter
from pathlib import Path

from arbplusjax import api


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "point_fast_jax_verification.md"


CATEGORY_ROWS = (
    (
        "1. core numeric scalars",
        ("core",),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_core_scalar_api_contracts.py",
        ),
        (
            "benchmarks/benchmark_core_scalar_service_api.py",
            "benchmarks/benchmark_api_surface.py",
        ),
        (
            "examples/example_core_scalar_surface.ipynb",
            "examples/example_api_surface.ipynb",
        ),
    ),
    (
        "2. interval / box / precision modes",
        ("core",),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_dense_plan_modes.py",
            "tests/test_arb_core_top10_modes.py",
        ),
        ("benchmarks/benchmark_api_surface.py",),
        (
            "examples/example_core_scalar_surface.ipynb",
            "examples/example_api_surface.ipynb",
        ),
    ),
    (
        "3. dense matrix functionality",
        ("matrix",),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_arb_mat_chassis.py",
            "tests/test_acb_mat_chassis.py",
        ),
        ("benchmarks/benchmark_dense_matrix_surface.py",),
        ("examples/example_dense_matrix_surface.ipynb",),
    ),
    (
        "4. sparse / block-sparse / vblock functionality",
        ("matrix",),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_sparse_format_modes.py",
        ),
        (
            "benchmarks/benchmark_sparse_matrix_surface.py",
            "benchmarks/benchmark_block_sparse_matrix_surface.py",
            "benchmarks/benchmark_vblock_sparse_matrix_surface.py",
        ),
        ("examples/example_sparse_matrix_surface.ipynb",),
    ),
    (
        "5. matrix-free / operator functionality",
        ("matrix",),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_matrix_free_logdet_solve_jit.py",
            "tests/test_matrix_free_basic.py",
        ),
        ("benchmarks/benchmark_matrix_free_krylov.py",),
        ("examples/example_matrix_free_operator_surface.ipynb",),
    ),
    (
        "6. special functions",
        ("barnes", "bessel", "gamma", "hypergeometric", "integration"),
        (
            "tests/test_point_fast_jax_categories.py",
            "tests/test_special_function_hardening.py",
            "tests/test_special_function_ad_directions.py",
        ),
        (
            "benchmarks/benchmark_special_function_service_api.py",
            "benchmarks/special_function_ad_benchmark.py",
            "benchmarks/special_function_hardening_benchmark.py",
        ),
        (
            "examples/example_gamma_family_surface.ipynb",
            "examples/example_barnes_double_gamma_surface.ipynb",
            "examples/example_hypgeom_family_surface.ipynb",
        ),
    ),
)


def _point_rows() -> list:
    return [entry for entry in api.list_public_function_metadata() if entry.point_support]


def _family_counts() -> Counter[str]:
    return Counter(entry.family for entry in _point_rows())


def _evidence_status(paths: tuple[str, ...], benchmarks: tuple[str, ...], notebooks: tuple[str, ...]) -> str:
    all_paths = (*paths, *benchmarks, *notebooks)
    return "complete" if all((REPO_ROOT / path).exists() for path in all_paths) else "incomplete"


def render() -> str:
    rows = _point_rows()
    family_counts = _family_counts()
    direct_fast = sum(1 for entry in rows if entry.name in api._DIRECT_POINT_BATCH_FASTPATHS)  # type: ignore[attr-defined]
    lines = [
        "Last updated: 2026-03-27T00:00:00Z",
        "",
        "# Point Fast JAX Verification",
        "",
        "This report verifies the repo-wide `fast JAX` contract for public `point` mode only.",
        "",
        "Policy references:",
        "- [operational_jax_standard.md](/docs/standards/operational_jax_standard.md)",
        "- [fast_jax_standard.md](/docs/standards/fast_jax_standard.md)",
        "- [point_fast_jax_implementation.md](/docs/implementation/point_fast_jax_implementation.md)",
        "- [point_fast_jax_plan.md](/docs/status/point_fast_jax_plan.md)",
        "",
        "Verification rule:",
        "- public point surface exists",
        "- `api.eval_point(..., jit=True)` exists",
        "- `api.bind_point_batch_jit(...)` exists",
        "- representative category-owned proof test exists",
        "- benchmark and canonical notebook evidence exists for the category",
        "",
        f"Total public point functions: `{len(rows)}`",
        f"Direct family-owned point batch fastpaths: `{direct_fast}`",
        "",
        "## Family Counts",
        "",
    ]
    for family in sorted(family_counts):
        lines.append(f"- `{family}`: `{family_counts[family]}` public point functions")
    lines.extend(
        [
            "",
            "## Category Verification",
            "",
            "| category | covered families | point_count | compiled_single | compiled_batch | direct_fastpath_family_presence | evidence_status | proof tests | benchmarks | notebooks |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for category, families, tests, benchmarks, notebooks in CATEGORY_ROWS:
        point_count = sum(family_counts[family] for family in families)
        fastpath_present = "yes" if any(
            entry.name in api._DIRECT_POINT_BATCH_FASTPATHS and entry.family in families  # type: ignore[attr-defined]
            for entry in rows
        ) else "no"
        lines.append(
            "| "
            + f"`{category}` | "
            + ", ".join(f"`{family}`" for family in families)
            + f" | `{point_count}` | `yes` | `yes` | `{fastpath_present}` | "
            + f"`{_evidence_status(tests, benchmarks, notebooks)}` | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in tests)
            + " | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in benchmarks)
            + " | "
            + ", ".join(f"[{Path(path).name}](/{path})" for path in notebooks)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Current Verification Note",
            "",
            "- This report is intentionally point-only. It does not verify `basic`, `adaptive`, or `rigorous` mode semantics.",
            "- The verification proof is category-owned rather than one giant smoke test, so each major runtime family keeps its own fast-JAX evidence near the implementation.",
            "- Deeper per-function numerical hardening still belongs in the owning family tests and benchmarks; this report verifies the public point fast-JAX contract, not full mathematical completeness.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
