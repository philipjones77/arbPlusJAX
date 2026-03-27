from tools import hypgeom_status_report as hsr


def test_hypgeom_status_report_surfaces_current_family_view() -> None:
    report = hsr.render()

    assert "Last updated: 2026-03-24T00:00:00Z" in report
    assert "canonical_four_mode_rows=" in report
    assert "alternative_four_mode_rows=" in report
    assert "The headline canonical families `0f1`, `1f1`, `2f1`, `u`, and `pfq`" in report
    assert "Orthogonal/classical families are now clearer in the view" in report
    assert "Alternative Boost and CuSF families now have a materially better view than before" in report
    assert "## Current Priority Gaps" in report
    assert "helper duplication in `hypgeom.py`" in report
