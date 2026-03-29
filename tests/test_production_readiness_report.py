from pathlib import Path

from tools import production_readiness_report as prr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_production_readiness_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "production_readiness.md"
    assert path.read_text(encoding="utf-8") == prr.render()


def test_production_readiness_report_covers_production_governance_areas() -> None:
    text = prr.render()
    for area in (
        "release and packaging",
        "docs publishing",
        "release governance",
        "security and supply chain",
        "operational support",
        "capability and maturity reporting",
    ):
        assert f"`{area}`" in text
    assert "build-dist.yml" in text
    assert "publish-release.yml" in text
    assert "docs-publish.yml" in text
    assert "dependency-audit.yml" in text
    assert "Production Readiness" in text
