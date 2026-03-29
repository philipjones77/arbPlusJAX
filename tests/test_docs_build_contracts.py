from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_build_structure_exists() -> None:
    required = [
        REPO_ROOT / "docs" / "conf.py",
        REPO_ROOT / "docs" / "README.md",
        REPO_ROOT / "docs" / "Makefile",
        REPO_ROOT / "docs" / "make.bat",
        REPO_ROOT / "tools" / "prepare_sphinx_docs.py",
        REPO_ROOT / "tools" / "build_docs.py",
        REPO_ROOT / ".github" / "workflows" / "docs-publish.yml",
    ]
    for path in required:
        assert path.exists(), f"Missing docs build artifact: {path}"


def test_docs_build_standard_exists() -> None:
    path = REPO_ROOT / "docs" / "standards" / "docs_build_standard.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "docs/README.md" in text
    assert "Sphinx + MyST" in text
    assert "GitHub Pages" in text
    assert "LuaLaTeX" in text
