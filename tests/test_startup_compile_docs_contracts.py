from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_startup_compile_standard_covers_repo_wide_policy_requirements() -> None:
    text = _read("docs/standards/startup_compile_standard.md")
    assert "stable-shape" in text
    assert "Centralized JIT ownership" in text
    assert "JAX_ENABLE_COMPILATION_CACHE=1" in text
    assert "JAX_COMPILATION_CACHE_DIR" in text
    assert "Warmup policy" in text
    assert "Process reuse policy" in text
    assert "Shared playbook and template" in text


def test_startup_compile_playbook_standard_exists_as_cross_repo_template_surface() -> None:
    text = _read("docs/standards/startup_compile_playbook_standard.md")
    assert "shared playbook" in text.lower()
    assert "pad_to" in text
    assert "Centralize JIT ownership" in text
    assert "Turn on persistent compilation cache everywhere" in text
    assert "Add compile budgets to CI" in text
    assert "Publish one shared playbook" in text


def test_startup_compile_repo_template_exists() -> None:
    text = _read("docs/implementation/startup_compile_repo_template.md")
    assert "Required Deliverables" in text
    assert "JAX_ENABLE_COMPILATION_CACHE=1" in text
    assert "JAX_COMPILATION_CACHE_DIR" in text
    assert "Required Probe Matrix" in text
    assert "Migration Sequence" in text
