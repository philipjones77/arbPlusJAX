from pathlib import Path

from tools import repo_standards_verification_report as rsvr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_standards_verification_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "repo_standards_verification.md"
    assert path.read_text(encoding="utf-8") == rsvr.render()


def test_repo_standards_verification_report_covers_runtime_compile_cache_and_release_surfaces() -> None:
    text = rsvr.render()
    for area in (
        "runtime/api",
        "caching/recompilation",
        "startup/import/compile",
        "point-only fast jax",
        "release/process/bootstrap",
    ):
        assert f"`{area}`" in text
    assert "cache_aware_surface_inventory.md" in text
    assert "api_cold_path_inventory.md" in text
    assert "point_fast_jax_verification.md" in text
    assert "point_basic_function_verification.md" in text
    assert "release_execution_checklist_standard.md" in text
