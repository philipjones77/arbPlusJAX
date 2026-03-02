from __future__ import annotations

import os
from pathlib import Path

import pytest


def _boost_root() -> Path:
    env = os.getenv("BOOST_MATH_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(r"C:\Users\phili\OneDrive\Documents\GitHub\math")


def test_boost_hypgeom_public_headers_and_fwd_signatures():
    root = _boost_root()
    inc = root / "include" / "boost" / "math" / "special_functions"
    if not inc.exists():
        pytest.skip(f"Boost.Math include path not found: {inc}")

    expected_headers = {
        "hypergeometric_0F1.hpp",
        "hypergeometric_1F0.hpp",
        "hypergeometric_1F1.hpp",
        "hypergeometric_2F0.hpp",
        "hypergeometric_pFq.hpp",
    }
    present = {p.name for p in inc.glob("hypergeometric*.hpp")}
    assert expected_headers.issubset(present)

    fwd = (inc / "math_fwd.hpp").read_text(encoding="utf-8", errors="ignore")
    for symbol in (
        "hypergeometric_1F0",
        "hypergeometric_0F1",
        "hypergeometric_2F0",
        "hypergeometric_1F1",
    ):
        assert symbol in fwd
