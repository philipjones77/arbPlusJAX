from pathlib import Path

from tools import report_status_refresh_inventory as rsri


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_report_status_refresh_inventory_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "report_status_refresh_inventory.md"
    assert path.read_text(encoding="utf-8") == rsri.render()


def test_every_report_and_status_markdown_file_is_listed() -> None:
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for folder in (REPO_ROOT / "docs" / "reports", REPO_ROOT / "docs" / "status")
        for path in folder.glob("*.md")
    }
    listed = {row[0] for row in rsri._status_rows()}
    assert listed == expected
