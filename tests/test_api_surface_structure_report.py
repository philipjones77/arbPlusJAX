from pathlib import Path

from tools import api_surface_structure_report as assr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_api_surface_structure_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "api_surface_structure.md"
    assert path.read_text(encoding="utf-8") == assr.render()


def test_api_surface_structure_report_covers_public_api_exports() -> None:
    text = assr.render()
    for name in (
        "evaluate",
        "eval_point",
        "bind_point_batch",
        "bind_point_batch_jit_with_diagnostics",
        "choose_point_batch_policy",
        "choose_matrix_free_plan_policy",
        "prewarm_matrix_free_kernels",
    ):
        assert f"`{name}`" in text
    assert "Unified Routing" in text
    assert "Bound Service Surfaces" in text
    assert "Compiled And AD Surfaces" in text
