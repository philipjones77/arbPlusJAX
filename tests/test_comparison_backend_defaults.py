from pathlib import Path
import sys

from tools import comparison_backend_defaults_report as cbdr


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from bench_registry import FUNCTIONS


def test_every_benchmark_function_has_a_default_external_reference() -> None:
    for spec in FUNCTIONS:
        assert (
            spec.default_interval_reference_backend()
            or spec.default_high_precision_reference_backend()
            or spec.default_float_reference_backend()
        ), f"{spec.name} is missing a default external comparison backend"


def test_default_reference_priority_is_ordered() -> None:
    for spec in FUNCTIONS:
        order = spec.default_comparison_backend_order()
        if spec.c_lib and spec.c_fn:
            assert order[0] == "c_arb"
        elif spec.mpmath:
            assert order[0] == "mpmath"
        elif spec.scipy:
            assert order[0] == "scipy"


def test_comparison_backend_defaults_report_is_current() -> None:
    path = REPO_ROOT / "docs" / "reports" / "comparison_backend_defaults.md"
    assert path.read_text(encoding="utf-8") == cbdr.render()
