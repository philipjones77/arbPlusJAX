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
        "example_dense_matrix_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_dirichlet_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
        "example_hypgeom_family_surface.ipynb",
    )
    for name in names:
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "## Production Pattern" in text
        assert "## Fast JAX Point Pattern" in text
        assert "## AD Product Pattern" in text
        assert "## Extending Benchmarks" in text


def test_canonical_notebooks_show_production_controls_or_caching() -> None:
    keyword_sets = {
        "example_core_scalar_surface.ipynb": ("bind_point_batch", "bind_point_batch_with_diagnostics", "choose_point_batch_policy", "prewarm_core_point_kernels"),
        "example_api_surface.ipynb": ("bind_point_batch", "bind_interval_batch", "bind_interval_batch_jit_with_diagnostics", "choose_interval_batch_policy", "cached_plan"),
        "example_dense_matrix_surface.ipynb": ("cached_matvec", "cached_rmatvec", "operator_plan"),
        "example_sparse_matrix_surface.ipynb": ("cached_prepare", "cached_apply", "pad_to"),
        "example_matrix_free_operator_surface.ipynb": ("plan_prepare", "preconditioner"),
        "example_fft_nufft_surface.ipynb": ("cached_prepare", "cached_apply"),
        "example_dirichlet_surface.ipynb": ("bind_point_batch_jit", "n_terms", "prec_bits"),
        "example_gamma_family_surface.ipynb": ("bind_point_batch", "bind_interval_batch", "bind_interval_batch_with_diagnostics"),
        "example_barnes_double_gamma_surface.ipynb": ("diagnostics", "prec_bits", "dps"),
        "example_hypgeom_family_surface.ipynb": ("bind_point_batch_jit", "bind_interval_batch", "bind_interval_batch_jit_with_diagnostics", "hypgeom_status", "special_function_hardening_benchmark"),
    }
    for name, keywords in keyword_sets.items():
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "bind_point_batch_jit" in text or "point_jit" in text or "cached_apply_jit" in text or "jax.jit(" in text
        for keyword in keywords:
            assert keyword in text, f"{name} missing {keyword}"


def test_canonical_notebooks_show_ad_validation_and_plotting() -> None:
    names = (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_dense_matrix_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_dirichlet_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
        "example_hypgeom_family_surface.ipynb",
    )
    for name in names:
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "jax.grad" in text or "jax.jvp" in text or "jax.jacfwd" in text or "jax.jacrev" in text
        assert "ad_validation_" in text
        assert "plot(" in text


def test_special_function_notebooks_show_argument_and_parameter_ad() -> None:
    for name in (
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
        "example_hypgeom_family_surface.ipynb",
    ):
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "parameter direction" in text.lower()
        assert "argument direction" in text.lower()
        assert "grad_" in text


def test_parameterized_surface_notebooks_show_argument_and_parameter_ad() -> None:
    for name in (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_dense_matrix_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
    ):
        text = _notebook_text(EXAMPLES_DIR / name)
        assert "parameter direction" in text.lower()
        assert "argument direction" in text.lower()
        assert "grad_" in text


def test_canonical_notebooks_expose_runtime_portability_contracts() -> None:
    names = (
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_dense_matrix_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_dirichlet_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
        "example_hypgeom_family_surface.ipynb",
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


def test_matrix_notebooks_cover_dense_sparse_block_vblock_and_operator_choices() -> None:
    dense_text = _notebook_text(EXAMPLES_DIR / "example_dense_matrix_surface.ipynb")
    sparse_text = _notebook_text(EXAMPLES_DIR / "example_sparse_matrix_surface.ipynb")
    matrix_free_text = _notebook_text(EXAMPLES_DIR / "example_matrix_free_operator_surface.ipynb")
    assert "cached_rmatvec" in dense_text
    assert "operator_plan" in dense_text
    assert "block_sparse" in sparse_text or "block" in sparse_text
    assert "vblock" in sparse_text
    assert "rmatvec" in sparse_text
    assert "sparse_operator_plan" in matrix_free_text
    assert "Compare" in matrix_free_text or "Contrast" in matrix_free_text
