from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text()


def test_heavy_modules_do_not_build_top_level_jit_aliases_eagerly():
    for relpath in (
        "src/arbplusjax/arb_core.py",
        "src/arbplusjax/acb_core.py",
        "src/arbplusjax/dirichlet.py",
        "src/arbplusjax/acb_dirichlet.py",
        "src/arbplusjax/nufft.py",
        "src/arbplusjax/jrb_mat.py",
        "src/arbplusjax/jcb_mat.py",
    ):
        text = _read(relpath)
        assert "= jax.jit(" not in text, relpath


def test_acb_core_defers_heavy_special_function_imports():
    text = _read("src/arbplusjax/acb_core.py")
    top_level = text.split("def _hypgeom():", 1)[0]
    assert "from . import acb_dirichlet" not in top_level
    assert "from . import barnesg" not in top_level
    assert "from . import hypgeom" not in top_level
    assert "def _hypgeom():" in text
    assert "def _barnesg():" in text
    assert "def _acb_dirichlet():" in text


def test_shared_lazy_jit_helper_exists():
    text = _read("src/arbplusjax/lazy_jit.py")
    assert "def lazy_jit(" in text


def test_acb_dirichlet_defers_series_missing_impl_import():
    text = _read("src/arbplusjax/acb_dirichlet.py")
    top_level = text.split("def __getattr__(name: str):", 1)[0]
    assert "from . import series_missing_impl as _smi" not in top_level
    assert "def __getattr__(name: str):" in text
