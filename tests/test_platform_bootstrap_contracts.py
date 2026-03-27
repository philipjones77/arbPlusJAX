from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_colab_requirements_file_exists_and_bootstrap_uses_it() -> None:
    requirements = REPO_ROOT / "requirements-colab.txt"
    bootstrap = REPO_ROOT / "tools" / "colab_bootstrap.sh"

    req_text = requirements.read_text(encoding="utf-8")
    bootstrap_text = bootstrap.read_text(encoding="utf-8")

    assert "-e ." in req_text
    assert "matplotlib" in req_text
    assert "REQUIREMENTS_FILE" in bootstrap_text
    assert "requirements-colab.txt" in bootstrap_text
    assert 'INSTALL_GPU_JAX="${INSTALL_GPU_JAX:-0}"' in bootstrap_text
    assert 'install_pkg -r "$REQUIREMENTS_FILE"' in bootstrap_text


def test_platform_bootstrap_profiles_cover_windows_linux_and_colab() -> None:
    config = REPO_ROOT / "configs" / "platform_bootstrap_profiles.json"
    payload = json.loads(config.read_text(encoding="utf-8"))

    assert set(payload) == {"linux", "wsl", "windows", "github_submission", "colab"}
    for key, entry in payload.items():
        assert entry["purpose"]
        assert entry["owner"]
        assert entry["execution_surface"]
        assert entry["backend_scope"] in {"CPU/GPU portable", "CPU-only", "backend-specific"}
        assert entry["bootstrap_command"]
        assert entry["validation_command"]


def test_portability_docs_reference_checked_in_bootstrap_surfaces() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    standard = (REPO_ROOT / "docs" / "standards" / "environment_portability_standard.md").read_text(encoding="utf-8")
    inventory = (REPO_ROOT / "docs" / "reports" / "environment_portability_inventory.md").read_text(encoding="utf-8")
    compare = (REPO_ROOT / "requirements-compare.txt").read_text(encoding="utf-8")
    config_readme = (REPO_ROOT / "configs" / "README.md").read_text(encoding="utf-8")

    for text in (readme, standard, inventory, config_readme):
        assert "requirements-colab.txt" in text
        assert "colab_bootstrap.sh" in text
        assert "optional_comparison_backends.json" in text

    assert "native Windows" in standard
    assert "platform_bootstrap_profiles.json" in inventory
    assert "mpmath" in compare
