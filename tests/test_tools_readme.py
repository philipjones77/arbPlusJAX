from pathlib import Path

from tools import generate_tools_readme as gtr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tools_readme_is_current() -> None:
    path = REPO_ROOT / "tools" / "README.md"
    assert path.read_text(encoding="utf-8") == gtr.render()


def test_tools_readme_prefers_canonical_tool_names() -> None:
    text = gtr.render()
    assert "make_zip.py" in text
    assert "update_docs_indexes.py" in text
    assert "removed" in text
    assert "MAKE_ZIP.py" in text
