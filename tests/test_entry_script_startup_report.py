from __future__ import annotations

from pathlib import Path

from tools import entry_script_startup_report as esr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_entry_script_startup_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "entry_script_startup_inventory.md"
    text = path.read_text(encoding="utf-8")
    assert "# Entry Script Startup Inventory" in text
    for script_path, _mode in esr.ENTRY_SCRIPTS:
        assert f"[{script_path}](/{script_path})" in text


def test_entry_script_startup_report_mentions_debt_and_backend_split() -> None:
    text = esr.render()
    assert "Remaining Top-Level `arbplusjax` Import Debt" in text
    assert "mostly dominated by JAX import/backend/runtime bootstrap cost" in text
