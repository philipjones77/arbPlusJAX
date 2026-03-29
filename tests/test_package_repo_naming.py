from pathlib import Path

import pytest

from tools import make_zip as mz


def test_default_output_path_uses_required_name(tmp_path: Path):
    root = tmp_path / "arbPlusJAX"
    root.mkdir()
    out = mz.resolve_output_path(root, None)
    assert mz.is_valid_source_zip_name(out.name, root.name)
    assert out.parent.name == "_bundles"


def test_custom_output_path_rejects_invalid_name(tmp_path: Path):
    root = tmp_path / "arbPlusJAX"
    root.mkdir()
    with pytest.raises(ValueError):
        mz.resolve_output_path(root, "dist/repo_snapshot.zip")


def test_custom_output_path_accepts_valid_name(tmp_path: Path):
    root = tmp_path / "arbPlusJAX"
    root.mkdir()
    out = mz.resolve_output_path(root, "dist/arbPlusJAX_source_2026-03-01.zip")
    assert out.name == "arbPlusJAX_source_2026-03-01.zip"
