from __future__ import annotations

import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "api_surface_structure.md"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from arbplusjax import api


_SECTION_ORDER = (
    "Unified Routing",
    "Direct Evaluation",
    "Specialized Public Function Surfaces",
    "Bound Service Surfaces",
    "Compiled And AD Surfaces",
    "Policy Helpers",
    "Metadata And Registry Surfaces",
    "Structured Payload Types",
)


def _section_for(name: str) -> str:
    if name == "evaluate":
        return "Unified Routing"
    if name.startswith("eval_"):
        return "Direct Evaluation"
    if name.startswith("bind_"):
        if name.endswith("_jit") or name.endswith("_jit_with_diagnostics") or name.endswith("_ad"):
            return "Compiled And AD Surfaces"
        return "Bound Service Surfaces"
    if name.startswith("choose_") or name.startswith("prewarm_"):
        return "Policy Helpers"
    if name.startswith(("list_", "get_", "render_")):
        return "Metadata And Registry Surfaces"
    if name.endswith(("Metadata", "Diagnostics", "Problem", "Policy")):
        return "Structured Payload Types"
    return "Specialized Public Function Surfaces"


def _kind_label(name: str) -> str:
    if name == "evaluate":
        return "auto-routing"
    if name.startswith("eval_"):
        return "direct"
    if name.startswith("bind_") and name.endswith("_with_diagnostics"):
        return "diagnostics-bearing binder"
    if name.startswith("bind_") and name.endswith("_jit"):
        return "compiled bound"
    if name.startswith("bind_") and name.endswith("_ad"):
        return "compiled AD binder"
    if name.startswith("bind_"):
        return "bound service"
    if name.startswith(("choose_", "prewarm_")):
        return "policy helper"
    if name.startswith(("list_", "get_", "render_")):
        return "metadata/registry"
    if name.endswith(("Metadata", "Diagnostics", "Problem", "Policy")):
        return "structured payload"
    return "specialized direct"


def _signature_string(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature unavailable)"


def _rows():
    rows = []
    for name in api.__all__:
        obj = getattr(api, name)
        rows.append(
            {
                "name": name,
                "section": _section_for(name),
                "kind": _kind_label(name),
                "signature": _signature_string(obj) if callable(obj) else "(type export)",
            }
        )
    return rows


def render() -> str:
    rows = _rows()
    section_counts: dict[str, int] = {}
    for row in rows:
        section_counts[row["section"]] = section_counts.get(row["section"], 0) + 1

    lines = [
        "Last updated: 2026-03-28T00:00:00Z",
        "",
        "# API Surface Structure",
        "",
        "This generated report consolidates the public `arbplusjax.api` surface into one place.",
        "",
        "Use it as the practical companion to:",
        "- [api_surface_kinds_standard.md](/docs/standards/api_surface_kinds_standard.md)",
        "- [jax_api_runtime_standard.md](/docs/standards/jax_api_runtime_standard.md)",
        "- [api_usability_standard.md](/docs/standards/api_usability_standard.md)",
        "",
        "Interpretation:",
        "- `direct` means an ordinary public evaluation surface.",
        "- `bound service` means a repeated-call binder or bound callable surface.",
        "- `compiled bound` means the repeated-call binder owns the compiled entrypoint explicitly.",
        "- `diagnostics-bearing binder` means the bound surface returns both value and structured diagnostics.",
        "- `policy helper` means the surface recommends or prepares the repeated-call policy rather than directly evaluating the function.",
        "",
        f"Total public `api` exports: `{len(rows)}`",
        "",
        "Section counts:",
    ]
    for section in _SECTION_ORDER:
        if section in section_counts:
            lines.append(f"- `{section}`: {section_counts[section]}")

    common_opts = {
        "bind_point_batch": ("dtype", "pad_to", "shape_bucket_multiple", "chunk_size", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_point_batch_with_diagnostics": ("dtype", "pad_to", "shape_bucket_multiple", "chunk_size", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_point_batch_jit": ("dtype", "pad_to", "shape_bucket_multiple", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_point_batch_jit_with_diagnostics": ("dtype", "pad_to", "shape_bucket_multiple", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_interval_batch": ("mode", "prec_bits", "dps", "dtype", "pad_to", "shape_bucket_multiple", "chunk_size", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_interval_batch_with_diagnostics": ("mode", "prec_bits", "dps", "dtype", "pad_to", "shape_bucket_multiple", "chunk_size", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_interval_batch_jit": ("mode", "prec_bits", "dps", "dtype", "pad_to", "shape_bucket_multiple", "backend", "min_gpu_batch_size", "prewarm"),
        "bind_interval_batch_jit_with_diagnostics": ("mode", "prec_bits", "dps", "dtype", "pad_to", "shape_bucket_multiple", "backend", "min_gpu_batch_size", "prewarm"),
        "choose_point_batch_policy": ("batch_size", "dtype", "backend", "pad_to", "shape_bucket_multiple", "chunk_size", "min_gpu_batch_size", "prewarm"),
        "choose_interval_batch_policy": ("batch_size", "dtype", "mode", "prec_bits", "dps", "backend", "pad_to", "shape_bucket_multiple", "chunk_size", "min_gpu_batch_size", "prewarm"),
        "prewarm_core_point_kernels": ("names", "dtype", "backend", "batch_size", "shape_bucket_multiple", "min_gpu_batch_size"),
        "prewarm_interval_mode_kernels": ("names", "dtype", "prec_bits", "dps", "backend", "batch_size", "shape_bucket_multiple", "min_gpu_batch_size"),
    }
    lines.extend(
        [
            "",
            "## Common Option Groups",
            "",
            "| surface | key options |",
            "|---|---|",
        ]
    )
    for name, opts in common_opts.items():
        lines.append(f"| `{name}` | `{', '.join(opts)}` |")

    for section in _SECTION_ORDER:
        section_rows = [row for row in rows if row["section"] == section]
        if not section_rows:
            continue
        lines.extend(
            [
                "",
                f"## {section}",
                "",
                "| public_name | kind | signature |",
                "|---|---|---|",
            ]
        )
        for row in section_rows:
            sig = row["signature"].replace("|", "\\|")
            lines.append(f"| `{row['name']}` | `{row['kind']}` | `{sig}` |")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
