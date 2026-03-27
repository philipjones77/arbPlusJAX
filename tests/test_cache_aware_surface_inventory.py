from pathlib import Path

from tools import cache_aware_surface_report as casr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cache_aware_surface_inventory_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "cache_aware_surface_inventory.md"
    assert path.read_text(encoding="utf-8") == casr.render()


def test_cache_aware_surface_inventory_lists_binders_examples_and_benchmarks() -> None:
    text = (REPO_ROOT / "docs" / "reports" / "cache_aware_surface_inventory.md").read_text(encoding="utf-8")
    for needle in (
        "api.bind_point_batch",
        "api.bind_point_batch_jit",
        "api.bind_interval_batch",
        "examples/example_sparse_matrix_surface.ipynb",
        "examples/example_hypgeom_family_surface.ipynb",
        "benchmarks/benchmark_sparse_matrix_surface.py",
        "benchmarks/special_function_hardening_benchmark.py",
    ):
        assert needle in text


def test_caching_recompilation_standard_points_to_generated_inventory() -> None:
    text = (REPO_ROOT / "docs" / "standards" / "caching_recompilation_standard.md").read_text(encoding="utf-8")
    assert "cache_aware_surface_inventory.md" in text
    assert "tools/cache_aware_surface_report.py" in text
