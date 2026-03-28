from pathlib import Path

import pytest

from tools import parameterized_ad_verification_report as padr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parameterized_ad_verification_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "parameterized_ad_verification.md"
    assert path.read_text(encoding="utf-8") == padr.render()


def test_parameterized_ad_verification_report_covers_special_function_matrix_and_curvature_rows() -> None:
    text = padr.render()
    for needle in (
        "arb_pow",
        "acb_hurwitz_zeta",
        "incomplete_gamma_upper",
        "incomplete_bessel_k",
        "double_gamma.ifj_barnesdoublegamma",
        "hypgeom.arb_hypgeom_1f1",
        "hypgeom.acb_hypgeom_u",
        "jrb_mat_multi_shift_solve_point",
        "curvature.make_posterior_precision_operator",
    ):
        assert f"`{needle}`" in text
    assert "`verified`" in text or "verified" in text


@pytest.mark.parametrize("case", padr.CASES, ids=lambda case: case.name)
def test_parameterized_public_cases_support_argument_and_parameter_ad(case) -> None:
    assert padr._is_finite_tree(case.argument_grad())
    assert padr._is_finite_tree(case.parameter_grad())
