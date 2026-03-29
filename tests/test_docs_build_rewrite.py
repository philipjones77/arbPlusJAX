import re
from pathlib import Path

from tools.prepare_sphinx_docs import prepare_docs


REPO_ROOT = Path(__file__).resolve().parents[1]
UNRESOLVED_DOC_LINK = re.compile(r"\[[^\]]+\]\(/docs/[^)]+\)")


def test_prepare_sphinx_docs_rewrites_repo_root_doc_links(tmp_path: Path) -> None:
    source_root = prepare_docs(tmp_path / "site-source")
    readme = source_root / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "[project_overview.md](project_overview.md)" in text

    for path in source_root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert not UNRESOLVED_DOC_LINK.search(text), f"Unresolved /docs/ markdown link in {path}"
