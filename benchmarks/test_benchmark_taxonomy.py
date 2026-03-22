from __future__ import annotations

from benchmarks.taxonomy import BENCHMARK_TAXONOMY
from benchmarks.taxonomy import discover_benchmark_scripts
from benchmarks.taxonomy import marker_names_for_script
from benchmarks.taxonomy import OFFICIAL_BENCHMARKS
from benchmarks.taxonomy import official_roles_for_script


def test_benchmark_taxonomy_covers_all_entrypoints() -> None:
    discovered = set(discover_benchmark_scripts())
    classified = set(BENCHMARK_TAXONOMY.keys())
    assert discovered == classified


def test_benchmark_taxonomy_derives_nonempty_marker_sets() -> None:
    for script_name in sorted(BENCHMARK_TAXONOMY):
        markers = marker_names_for_script(script_name)
        assert "benchmark" in markers
        assert any(name.startswith("benchmark_") and name != "benchmark" for name in markers)


def test_official_benchmark_targets_exist_in_taxonomy() -> None:
    classified = set(BENCHMARK_TAXONOMY.keys())
    assert OFFICIAL_BENCHMARKS
    for target in OFFICIAL_BENCHMARKS.values():
        assert target in classified


def test_official_benchmarks_receive_official_marker() -> None:
    for script_name in sorted(BENCHMARK_TAXONOMY):
        markers = marker_names_for_script(script_name)
        roles = official_roles_for_script(script_name)
        if roles:
            assert "benchmark_official" in markers
