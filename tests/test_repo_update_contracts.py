from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_update_docs_exist() -> None:
    assert (REPO_ROOT / "docs" / "standards" / "update_standard.md").exists()
    assert (REPO_ROOT / "docs" / "objects" / "update_artifacts.md").exists()
    assert (REPO_ROOT / "docs" / "implementation" / "update_implementation.md").exists()


def test_tools_readme_mentions_update_entrypoints() -> None:
    text = _read("tools/README.md")
    assert "python tools/update_repo_artifacts.py" in text
    assert "python tools/check_repo_update_drift.py" in text


def test_update_standard_points_to_expected_doc_layers() -> None:
    text = _read("docs/standards/update_standard.md")
    assert "docs/standards/" in text
    assert "docs/objects/" in text
    assert "docs/implementation/" in text
    assert "tools/" in text
    assert "tests/" in text
