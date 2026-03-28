from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_release_execution_checklist_standard_covers_repo_release_requirements() -> None:
    text = _read("docs/standards/release_execution_checklist_standard.md")
    assert "release-quality" in text
    assert "CPU owner-test slice" in text
    assert "startup or first-use probe slice" in text
    assert "implicit-adjoint" in text
    assert "parameter-direction AD coverage" in text
    assert "generated artifact refresh" in text


def test_release_execution_checklist_runbook_exists() -> None:
    text = _read("docs/implementation/release_execution_checklist.md")
    assert "Required Deliverables" in text
    assert "Minimum Command Matrix" in text
    assert "python tools/update_repo_artifacts.py" in text
    assert "python tools/check_repo_update_drift.py" in text
    assert "python tools/run_example_notebooks.py" in text

