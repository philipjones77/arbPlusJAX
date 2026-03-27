from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_tests_workflow_covers_windows_ubuntu_colab_and_compare_stack() -> None:
    text = _read(".github/workflows/tests.yml")

    assert "ubuntu-latest" in text
    assert "windows-latest" in text
    assert "tools/run_test_harness.py --profile chassis --jax-mode cpu" in text
    assert "tools/run_test_harness.py --profile bench-smoke --jax-mode cpu" in text
    assert "bash tools/colab_bootstrap.sh \"$PWD\"" in text
    assert "requirements-compare.txt" in text
    assert "tests/test_comparison_backend_defaults.py" in text


def test_docs_workflow_validates_generated_artifacts_without_auto_push() -> None:
    text = _read(".github/workflows/docs-indexes.yml")

    assert "python tools/check_generated_reports.py" in text
    assert "git diff --exit-code" in text
    assert "git push" not in text


def test_optional_backend_policy_is_declared_in_config_and_docs() -> None:
    config_text = _read("configs/optional_comparison_backends.json")
    readme = _read("README.md")
    inventory = _read("docs/reports/environment_portability_inventory.md")
    standard = _read("docs/standards/environment_portability_standard.md")

    for token in ("c_arb", "mathematica", "mpmath", "scipy", "jax_scipy", "experimental_jax"):
        assert token in config_text

    for text in (readme, inventory, standard):
        assert "optional_comparison_backends.json" in text
        assert "Mathematica" in text
        assert "jax.scipy" in text
