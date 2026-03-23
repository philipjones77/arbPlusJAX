from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

from bench_registry import FUNCTIONS


OUT_PATH = REPO_ROOT / "docs" / "reports" / "comparison_backend_defaults.md"


def render() -> str:
    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Comparison Backend Defaults",
        "",
        "This report records the default comparison stack for each benchmark-harness function.",
        "",
        "Policy:",
        "- use `c_arb` as the default interval/enclosure reference whenever a C Arb/FLINT adapter exists",
        "- use `mpmath` as the default high-precision point reference whenever available",
        "- use `scipy` as the default float64 engineering parity reference whenever available",
        "- keep `jax_point` as an internal point-mode comparison path rather than the primary external truth source",
        "",
        "| function | interval default | high-precision default | float default | comparison order |",
        "|---|---|---|---|---|",
    ]
    for spec in FUNCTIONS:
        defaults = spec.comparison_backend_fields()
        order = ", ".join(f"`{name}`" for name in spec.default_comparison_backend_order()) or "`none`"
        lines.append(
            f"| `{spec.name}` | `{defaults['interval'] or 'none'}` | `{defaults['high_precision'] or 'none'}` | `{defaults['float'] or 'none'}` | {order} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
