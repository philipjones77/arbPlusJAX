from pathlib import Path

from tools import point_fast_jax_category_report as pfjcr


REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_CATEGORIES = (
    "1. core numeric scalars",
    "2. interval / box / precision modes",
    "3. dense matrix functionality",
    "4. sparse / block-sparse / vblock functionality",
    "5. matrix-free / operator functionality",
    "6. special functions",
)


def test_point_fast_jax_category_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "point_fast_jax_category_matrix.md"
    assert path.read_text(encoding="utf-8") == pfjcr.render()


def test_fast_jax_standard_covers_all_six_categories() -> None:
    text = (REPO_ROOT / "docs" / "standards" / "fast_jax_standard.md").read_text(encoding="utf-8").lower()
    for category in EXPECTED_CATEGORIES:
        assert category in text


def test_point_fast_jax_plan_covers_all_six_categories() -> None:
    text = (REPO_ROOT / "docs" / "status" / "point_fast_jax_plan.md").read_text(encoding="utf-8").lower()
    for category in EXPECTED_CATEGORIES:
        assert category in text


def test_point_fast_jax_implementation_covers_all_six_categories() -> None:
    text = (REPO_ROOT / "docs" / "implementation" / "point_fast_jax_implementation.md").read_text(encoding="utf-8").lower()
    for category in EXPECTED_CATEGORIES:
        assert category in text
