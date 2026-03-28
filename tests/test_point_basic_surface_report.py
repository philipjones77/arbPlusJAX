from pathlib import Path

from tools import point_basic_surface_report as pbsr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_basic_surface_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "point_basic_surface_status.md"
    assert path.read_text(encoding="utf-8") == pbsr.render()


def test_point_basic_surface_report_covers_public_point_basic_families_and_curvature() -> None:
    text = pbsr.render()
    for family in ("barnes", "bessel", "core", "gamma", "hypergeometric", "integration", "matrix"):
        assert f"`{family}`" in text
    assert "`curvature`" in text
    assert "point_count" in text
    assert "basic_count" in text
    assert "diagnostics_count" in text
    assert "ad_status" in text
    assert "argument+parameter" in text
