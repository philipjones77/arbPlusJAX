from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_high_priority_example_notebooks_exist() -> None:
    required = {
        "example_core_scalar_surface.ipynb",
        "example_api_surface.ipynb",
        "example_dense_matrix_surface.ipynb",
        "example_sparse_matrix_surface.ipynb",
        "example_matrix_free_operator_surface.ipynb",
        "example_fft_nufft_surface.ipynb",
        "example_dirichlet_surface.ipynb",
        "example_gamma_family_surface.ipynb",
        "example_barnes_double_gamma_surface.ipynb",
    }
    existing = {path.name for path in EXAMPLES_DIR.glob("example_*.ipynb")}
    assert required.issubset(existing)
