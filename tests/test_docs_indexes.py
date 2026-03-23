from pathlib import Path

from tools import generate_docs_indexes as gdi


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generated_docs_indexes_are_current() -> None:
    for path, content in gdi.generated_docs().items():
        assert path.read_text(encoding="utf-8") == content, f"Generated docs index out of date: {path}"
