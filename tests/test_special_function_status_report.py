from pathlib import Path

from tools import special_function_status_report as sfsr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_special_function_status_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "special_function_status.md"
    assert path.read_text(encoding="utf-8") == sfsr.render()


def test_special_function_status_report_lists_special_artifacts_and_examples() -> None:
    text = (REPO_ROOT / "docs" / "reports" / "special_function_status.md").read_text(encoding="utf-8")
    for needle in (
        "special_function_hardening_benchmark.py",
        "hypgeom_point_startup_probe.py",
        "double_gamma_point_startup_probe.py",
        "special_function_ad_benchmark.py",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
        "example_hypgeom_family_surface.ipynb",
        "ifj_barnesdoublegamma_diagnostics",
        "incomplete_bessel_i",
        "hypgeom.arb_hypgeom_1f1",
        "tests/test_special_function_ad_directions.py",
    ):
        assert needle in text
