from __future__ import annotations

from pathlib import Path


CANONICAL_KINDS = {
    "chassis",
    "parity",
    "contracts",
    "modes",
    "metadata",
    "diagnostics",
    "api",
    "surface",
    "smoke",
    "adjoints",
    "special",
    "hardening",
    "manifest",
    "naming",
}

TRANSITIONAL_KINDS = {
    "adapter",
    "aliases",
    "basic",
    "contract",
    "compat",
    "completeness",
    "complete",
    "engineering",
    "gamma",
    "hypgeom",
    "impls",
    "inventory",
    "inverse",
    "i",
    "k",
    "kernels",
    "layer",
    "mode",
    "new",
    "ops",
    "precision",
    "reports",
    "scaffold",
    "status",
    "tail",
    "tier1",
    "updates",
    "wrappers",
}

ALLOWED_KINDS = CANONICAL_KINDS | TRANSITIONAL_KINDS
SINGLETON_FAMILIES = {
    "boost_hypgeom",
    "incomplete_gamma",
    "nufft",
    "shahen_double_gamma",
}


def test_test_filenames_use_known_kind_tokens():
    test_files = sorted(Path(__file__).parent.glob("test_*.py"))
    unknown: list[str] = []
    malformed: list[str] = []

    for path in test_files:
        stem = path.stem
        parts = stem.split("_")
        if len(parts) == 2 and parts[1] in SINGLETON_FAMILIES:
            continue
        if len(parts) < 3:
            malformed.append(path.name)
            continue
        kind = parts[-1]
        if kind not in ALLOWED_KINDS:
            unknown.append(path.name)

    assert not malformed, f"malformed test filenames: {malformed}"
    assert not unknown, f"unknown test kind tokens: {unknown}"


def test_test_filenames_start_with_test_prefix():
    test_files = sorted(Path(__file__).parent.glob("*.py"))
    allowed_non_test_files = {"__init__.py", "conftest.py"}
    offenders = [
        path.name
        for path in test_files
        if path.name not in allowed_non_test_files and not path.name.startswith("test_") and not path.name.startswith("_")
    ]
    assert not offenders, f"test files must start with test_: {offenders}"
