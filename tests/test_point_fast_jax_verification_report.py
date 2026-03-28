from pathlib import Path

from tools import point_fast_jax_verification_report as pfjvr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_fast_jax_verification_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "point_fast_jax_verification.md"
    assert path.read_text(encoding="utf-8") == pfjvr.render()


def test_point_fast_jax_verification_report_covers_all_six_categories() -> None:
    text = pfjvr.render()
    for category in (
        "1. core numeric scalars",
        "2. interval / box / precision modes",
        "3. dense matrix functionality",
        "4. sparse / block-sparse / vblock functionality",
        "5. matrix-free / operator functionality",
        "6. special functions",
    ):
        assert f"`{category}`" in text
    assert "compiled_single" in text
    assert "compiled_batch" in text
