from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _notebook_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", ()):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.extend(source)
        else:
            chunks.append(source)
    return "\n".join(chunks)


def test_canonical_notebooks_include_production_pattern_section() -> None:
    names = (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
    )
    for name in names:
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "## Production Pattern" in text
        assert "## AD Product Pattern" in text
        assert "## Extending Benchmarks" in text


def test_canonical_notebooks_show_production_controls_or_caching() -> None:
    keyword_sets = {
        "example_core_scalar_surface.ipynb": ("bind_point_batch", "pad_to"),
        "example_api_surface.ipynb": ("bind_point_batch", "bind_interval_batch", "cached_plan"),
        "example_sparse_matrix_surface.ipynb": ("cached_prepare", "cached_apply", "pad_to"),
        "example_matrix_free_operator_surface.ipynb": ("plan_prepare", "preconditioner"),
        "example_fft_nufft_surface.ipynb": ("cached_prepare", "cached_apply"),
        "example_gamma_family_surface.ipynb": ("bind_point_batch", "bind_interval_batch"),
        "example_barnes_double_gamma_surface.ipynb": ("diagnostics", "prec_bits", "dps"),
    }
    for name, keywords in keyword_sets.items():
        text = _notebook_text(EXAMPLES_DIR / name)
        for keyword in keywords:
            assert keyword in text, f"{name} missing {keyword}"


def test_canonical_notebooks_show_ad_validation_and_plotting() -> None:
    names = (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
    )
    for name in names:
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "jax.grad" in text or "jax.jvp" in text
        assert "ad_validation_" in text
        assert "plot(" in text


def test_canonical_notebooks_expose_runtime_portability_contracts() -> None:
    names = (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
    )
    for name in names:
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "JAX_MODE" in text
        assert "JAX_DTYPE" in text
        assert "float32" in text
        assert "float64" in text
        assert "cpu" in text
        assert "gpu" in text
        assert "validation_slice" in text
