from __future__ import annotations

from pathlib import Path

from arbplusjax import api


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "docs" / "reports" / "point_fast_jax_function_inventory.md"


def _status_row(entry) -> tuple[str, str, str, str, str, str, str, str]:
    direct_batch_fastpath = "yes" if entry.name in api._DIRECT_POINT_BATCH_FASTPATHS else "no"  # type: ignore[attr-defined]
    if direct_batch_fastpath == "yes":
        hot_path = "direct batch kernel"
    else:
        hot_path = "compiled vmap fallback"
    return (
        entry.name,
        entry.family,
        entry.module,
        entry.stability,
        "yes",  # eval_point(..., jit=True)
        "yes",  # bind_point_batch_jit(...)
        direct_batch_fastpath,
        hot_path,
    )


def render() -> str:
    rows = [_status_row(entry) for entry in api.list_public_function_metadata() if entry.point_support]
    family_counts: dict[str, int] = {}
    for _, family, *_rest in rows:
        family_counts[family] = family_counts.get(family, 0) + 1

    lines = [
        "Last updated: 2026-03-23T00:00:00Z",
        "",
        "# Point Fast JAX Function Inventory",
        "",
        "This report tracks the full public `point` registry against the repo-wide fast-JAX public-surface contract.",
        "",
        "Interpretation:",
        "- `compiled_single` means the function is available through `api.eval_point(..., jit=True)`.",
        "- `compiled_batch` means the function is available through `api.bind_point_batch_jit(...)`.",
        "- `direct_batch_fastpath` means the batched surface has a family-owned direct batch kernel registered in the API.",
        "- `compiled vmap fallback` still satisfies the public compiled point-batch contract, but indicates a generic API batch layer rather than a family-owned direct batch kernel.",
        "",
        "Family counts:",
    ]
    for family in sorted(family_counts):
        lines.append(f"- `{family}`: {family_counts[family]}")
    lines.extend(
        [
            "",
            f"Total public point functions: `{len(rows)}`",
            "",
            "| public_name | family | module | stability | compiled_single | compiled_batch | direct_batch_fastpath | public hot-path status |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, family, module, stability, compiled_single, compiled_batch, direct_batch_fastpath, hot_path in rows:
        lines.append(
            f"| `{name}` | `{family}` | `{module}` | `{stability}` | `{compiled_single}` | `{compiled_batch}` | `{direct_batch_fastpath}` | {hot_path} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
