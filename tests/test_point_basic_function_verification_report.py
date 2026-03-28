from pathlib import Path

from tools import point_basic_function_verification_report as pbfvr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_basic_function_verification_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "point_basic_function_verification.md"
    assert path.read_text(encoding="utf-8") == pbfvr.render()


def test_point_basic_function_verification_report_covers_public_point_basic_rows() -> None:
    text = pbfvr.render()
    assert "verification_status" in text
    assert "direct_tested" in text
    assert "argument+parameter" in text
    assert "`verified`" in text or "verified" in text
    for function_name in ("arb_abs", "acb_dirichlet_zeta", "acb_mat_add", "incomplete_gamma_upper"):
        assert f"`{function_name}`" in text
