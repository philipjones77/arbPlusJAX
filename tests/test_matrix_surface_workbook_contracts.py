from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS = _REPO_ROOT / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from benchmarks import matrix_surface_workbook


REPO_ROOT = _REPO_ROOT


def test_matrix_surface_workbook_mentions_all_matrix_families_and_comparison_guidance() -> None:
    text = matrix_surface_workbook.render(n=4, warmup=0, runs=1, steps=4)
    assert "Dense Matrix Surface" in text
    assert "Sparse Matrix Surface" in text
    assert "Block Sparse Matrix Surface" in text
    assert "Variable-Block Sparse Matrix Surface" in text
    assert "Matrix-Free Surface" in text
    assert "Compare and Contrast" in text
    assert "cached matvec/rmatvec" in text
    assert "Recommended visualizations" in text
