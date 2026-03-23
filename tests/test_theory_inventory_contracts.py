from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = REPO_ROOT / "docs" / "theory"


def test_theory_index_references_current_methodology_notes() -> None:
    readme = (THEORY_DIR / "README.md").read_text(encoding="utf-8")
    for name in (
        "ball_arithmetic_and_modes.md",
        "matrix_interval_and_modes.md",
        "core_functions_methodology.md",
        "bessel_family_methodology.md",
        "gamma_family_methodology.md",
        "transform_fft_nufft_methodology.md",
    ):
        assert name in readme
