from pathlib import Path

from arbplusjax import api
from tools import point_fast_jax_function_report as pfjfr


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_point_fast_jax_function_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "point_fast_jax_function_inventory.md"
    assert path.read_text(encoding="utf-8") == pfjfr.render()


def test_every_public_point_function_is_listed_with_compiled_surfaces() -> None:
    text = pfjfr.render()
    for entry in api.list_public_function_metadata():
        if not entry.point_support:
            continue
        assert f"`{entry.name}`" in text
    assert "compiled_single" in text
    assert "compiled_batch" in text


def test_public_point_registry_has_universal_compiled_api_surfaces() -> None:
    point_entries = [entry for entry in api.list_public_function_metadata() if entry.point_support]
    rows = [pfjfr._status_row(entry) for entry in point_entries]
    assert rows
    assert all(row[4] == "yes" for row in rows)
    assert all(row[5] == "yes" for row in rows)
    assert any(row[6] == "yes" for row in rows)
    assert any(row[6] == "no" for row in rows)
